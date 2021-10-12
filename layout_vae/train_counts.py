import os
import sys
import random
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

from count import AutoregressiveCountEncoder, AutoregressiveCountDecoder
from layout import BatchCollator, LayoutDataset


def evaluate(model, loader, loss):
    errors = []
    model.eval()
    losses = None
    count_losses = []
    divergence_losses = []

    for batch_i, (indexes, target) in tqdm(enumerate(loader)):
        label_set = torch.stack([t.label_set for t in target], dim=0).float().to(device)
        counts = torch.stack([t.count for t in target], dim=0).float().to(device)
        batch_size = label_set.size(0)

        label_set_size = torch.sum(label_set > 0, dim=1).float()

        # depending on forcing, this can be prediction or ground truth.
        previous_counts = torch.zeros((batch_size, NUMBER_LABELS)).to(device)
        predicted_counts = torch.zeros((batch_size, NUMBER_LABELS)).to(device)

        batch_errors = torch.zeros((NUMBER_LABELS,), device=device).to(device)
        for label_i in range(NUMBER_LABELS):
            current_count = counts[:, label_i]
            has_nonzero = current_count > 0
            nonzero_batch_size = has_nonzero.nonzero().size(0)

            if nonzero_batch_size > 0:
                current_label_set = label_set[has_nonzero, :]
                current_previous_counts = previous_counts[has_nonzero, :]
                current_count = current_count[has_nonzero].unsqueeze(-1) #.unsqueeze(-1)
                current_label = label_encodings[label_i].unsqueeze(0).repeat(nonzero_batch_size, 1)

                log_rate, kl_divergence, _ = model(current_count, current_label_set, current_label, current_previous_counts)

                current_loss = loss(log_rate, current_count)
                losses = current_loss if losses is None else torch.cat([losses, current_loss])

                count_loss_i = count_loss(log_rate, current_count)
                # print(count_loss_i)
                count_losses.append(count_loss_i.reshape(-1))
                divergence_losses.append(kl_divergence.reshape(-1))

                # # predict the counts (try the lame way for now)
                # predicted_count = []

                # for nz_i in range(nonzero_batch_size):
                #     rate_i = torch.exp(log_rate[nz_i])
                #     dist = Poisson(rate_i)
                #     predicted_count_i = dist.sample((1,))[0] + 1
                #     predicted_count.append(predicted_count_i)

                # predicted_count = torch.cat(predicted_count, dim=0)
                # I think technically this needs to take care of the integer case.
                predicted_count = torch.floor(torch.exp(torch.squeeze(log_rate, dim=-1))) + 1
                predicted_counts[has_nonzero, label_i] = predicted_count

                batch_errors[label_i] = torch.mean(torch.abs(torch.squeeze(current_count, dim=-1) - predicted_count))

                # teacher forcing when evaluating reconstructions?
                previous_counts_mask = torch.cat((
                    torch.ones((batch_size, label_i + 1), device=device),
                    torch.zeros((batch_size, NUMBER_LABELS - label_i - 1),
                    device=device)), dim=-1)
                previous_counts = previous_counts_mask * counts

        errors.append(batch_errors)

    errors = torch.stack(errors, dim=0)
    average_error = torch.mean(errors, dim=0)
    average_loss = torch.mean(losses)
    print(f"validation: average error per class: {average_error}")
    print(f"validation: average loss: {average_loss}")
    count_losses = torch.cat(count_losses)
    divergence_losses = torch.cat(divergence_losses)
    loss_epoch = torch.mean(count_losses) + torch.mean(divergence_losses)

    return loss_epoch.item()


class PoissonLogLikelihood(nn.Module):
    def __init__(self):
        super(PoissonLogLikelihood, self).__init__()

    def forward(self, log_rate, count):
        # learned over count - 1?
        count = count - 1
        log_factorial = torch.lgamma(count + 1)
        log_likelihood = -torch.exp(log_rate) + count * log_rate - log_factorial

        # I assume this will be like N x [max # labels]
        return -log_likelihood


class AutoregressiveVariationalAutoencoder(nn.Module):
    def __init__(self, number_labels, conditioning_size, representation_size):
        super(AutoregressiveVariationalAutoencoder, self).__init__()

        self.representation_size = representation_size

        self.encoder = AutoregressiveCountEncoder(number_labels, conditioning_size, representation_size)
        self.decoder = AutoregressiveCountDecoder(conditioning_size, representation_size)

    def sample(self, mu, log_var):
        batch_size = mu.size(0)
        device = mu.device

        standard_normal = torch.randn((batch_size, self.representation_size), device=device)
        z = mu + standard_normal * torch.exp(0.5 * log_var)

        kl_divergence = -0.5 * torch.sum(
            1 + log_var - (mu ** 2) - torch.exp(log_var), dim=1)

        return z, kl_divergence

    def forward(self, x, label_set, current_label, count_so_far):
        mu, s, condition = self.encoder(x, label_set, current_label, count_so_far)
        z, kl_divergence = self.sample(mu, s)
        log_rate = self.decoder(z, condition)

        return log_rate, kl_divergence, z


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Count VAE')
    parser.add_argument("--exp", default="count_vae", help="postfix for experiment name")
    parser.add_argument("--log_dir", default="./logs", help="/path/to/logs/dir")
    parser.add_argument("--train_json", default="./instances_train.json", help="/path/to/train/json")
    parser.add_argument("--val_json", default="./instances_val.json", help="/path/to/val/json")

    parser.add_argument("--max_length", type=int, default=128, help="batch size")

    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--beta_1", type=float, default=0.9, help="beta_1 for adam")
    parser.add_argument('--evaluate', action='store_true', help="evaluate only")
    parser.add_argument('--save_every', type=int, default=10, help="evaluate only")

    args = parser.parse_args()

    if not args.evaluate:
        now = datetime.now().strftime("%m%d%y_%H%M%S")
        log_dir = os.path.join(args.log_dir, f"{now}_{args.exp}")
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        log_dir = args.log_dir
        ckpt_dir = os.path.join(log_dir, "checkpoints")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    collator = BatchCollator()
    train_dataset = LayoutDataset(args.train_json, args.max_length)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator)

    validation_dataset = LayoutDataset(args.val_json, args.max_length)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator)

    NUMBER_LABELS = train_dataset.number_labels

    label_encodings = torch.eye(NUMBER_LABELS).float().to(device)
    count_loss = PoissonLogLikelihood().to(device)

    autoencoder = AutoregressiveVariationalAutoencoder(
        NUMBER_LABELS,
        conditioning_size=128,
        representation_size=32).to(device)

    # evaluate the model
    if args.evaluate:
        min_epoch = -1
        min_loss = 1e100
        for epoch in range(args.epochs):
            checkpoint_path = os.path.join(log_dir, "checkpoints", 'epoch_%d.pth' % epoch)
            if not os.path.exists(checkpoint_path):
                continue
            print('Evaluating', checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            autoencoder.load_state_dict(checkpoint["model_state_dict"], strict=True)
            loss = evaluate(autoencoder, validation_loader, count_loss)
            print('End of epoch %d : %f' % (epoch, loss))
            if loss < min_loss:
                min_loss = loss
                min_epoch = epoch
        print('Best epoch: %d Best nll: %f' % (min_epoch, min_loss))
        sys.exit(0)

    opt = optim.Adam(autoencoder.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))
    epoch_number = 0
    while True:
        if (epoch_number > 0) and (epoch_number == args.epochs):
            print("done!")
            break

        print(f"starting epoch {epoch_number+1}")
        autoencoder.train()

        with tqdm(enumerate(train_loader)) as tq:
            for batch_i, (indexes, target) in tq:
                autoencoder.zero_grad()
                count_loss.zero_grad()

                label_set = torch.stack([t.label_set for t in target], dim=0).float().to(device)
                counts = torch.stack([t.count for t in target], dim=0).float().to(device)

                batch_size = label_set.size(0)

                label_set_size = torch.sum(label_set > 0, dim=1).float()

                count_losses = []
                divergence_losses = []

                previous_counts = torch.zeros((batch_size, NUMBER_LABELS)).to(device)

                for label_i in range(NUMBER_LABELS):
                    current_count_loss = torch.zeros((batch_size,)).to(device)
                    current_divergence_loss = torch.zeros((batch_size,)).to(device)

                    current_count = counts[:, label_i]
                    has_nonzero = current_count > 0
                    nonzero_batch_size = has_nonzero.nonzero().size(0)
                    if nonzero_batch_size > 0:
                        current_label_set = label_set[has_nonzero, :]
                        current_previous_counts = previous_counts[has_nonzero, :]
                        current_count = current_count[has_nonzero].unsqueeze(-1)  # .unsqueeze(-1)
                        current_label = label_encodings[label_i].unsqueeze(0).repeat(nonzero_batch_size, 1)

                        log_rate, kl_divergence, z = autoencoder(current_count, current_label_set, current_label,
                                                                 current_previous_counts)
                        count_loss_i = count_loss(log_rate, current_count)
                        current_count_loss[has_nonzero] = count_loss_i[:, 0]
                        count_losses.append(current_count_loss)

                        current_divergence_loss[has_nonzero] = kl_divergence
                        divergence_losses.append(current_divergence_loss)
                        # unsure if we do backward() here?

                    # teacher forcing!
                    previous_counts_mask = torch.cat((
                        torch.ones((batch_size, label_i + 1), device=device),
                        torch.zeros((batch_size, NUMBER_LABELS - label_i - 1), device=device)), dim=-1)
                    previous_counts = previous_counts_mask * counts

                count_losses = torch.stack(count_losses, dim=-1)
                count_loss_batch = torch.mean(torch.sum(count_losses, dim=-1) / label_set_size)

                divergence_losses = torch.stack(divergence_losses, dim=-1)
                divergence_loss_batch = torch.mean(torch.sum(divergence_losses, dim=-1) / label_set_size)

                loss_batch = count_loss_batch + 0.01 * divergence_loss_batch
                loss_batch.backward()
                opt.step()

                tq.set_description(f"{epoch_number+1}/{args.epochs} count_loss: {count_loss_batch.item()} "
                                  f"divergence_loss: {divergence_loss_batch.item()}")

        evaluate(autoencoder, validation_loader, count_loss)
        # print(f"validation loss [{epoch_number}/{args.epochs}: {validation_loss.item():4f}")

        # write out a checkpoint too.
        if (epoch_number + 1) % args.save_every == 0:
            torch.save({
                "epoch": epoch_number,
                "model_state_dict": autoencoder.state_dict(),
            }, os.path.join(ckpt_dir, "epoch_{0}.pth".format(epoch_number)))

        epoch_number += 1
