#!/usr/bin/env python3

"""
Bonito training with sparsity preservation using external masks.
"""

import os
import re
import glob
import math
import pickle
import sys
from pathlib import Path
from importlib import import_module
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from itertools import islice
from time import perf_counter
from collections import OrderedDict
from datetime import datetime

import toml
import torch
import numpy as np
from tqdm import tqdm
import torch.cuda.amp as amp

from bonito.training import linear_warmup_cosine_decay
from bonito.data import load_data, ModelSetup, ComputeSettings, DataSettings
from bonito.util import __models_dir__, default_config, load_model, load_symbol, init
from bonito.io import CSVLogger
from bonito.util import accuracy, decode_ref, permute, match_names, tqdm_environ


class ClipGrad:
    def __init__(self, quantile=0.5, factor=2.0, buffer_size=100):
        self.buffer = np.full(buffer_size, fill_value=1e6)
        self.quantile = quantile
        self.factor = factor
        self.i = 0

    def append(self, grad_norm):
        self.buffer[self.i] = grad_norm
        self.i = (self.i + 1) % len(self.buffer)

    def __call__(self, parameters):
        max_norm = self.factor * np.quantile(self.buffer, self.quantile)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm).item()
        if not math.isnan(grad_norm):
            self.append(grad_norm)
        return grad_norm


def load_state(dirname, device, model, optim=None, masks_path=None):
    model.to(device)
    if hasattr(model, "module"):
        model = model.module

    weight_no = optim_no = None
    optim_files = glob.glob(os.path.join(dirname, "optim_*.tar"))
    optim_nos = {int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in optim_files}

    weight_files = glob.glob(os.path.join(dirname, "weights_*.tar"))
    weight_nos = {int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files}

    to_load = []
    if optim is not None:
        weight_no = optim_no = max(optim_nos & weight_nos, default=None)
    else:
        weight_no = max(weight_nos, default=None)

    if weight_no:
        to_load.append(("weights", model))
    if optim_no:
        to_load.append(("optim", optim))

    if to_load:
        print(f"[picking up {', '.join([n for n, _ in to_load])} state from epoch {weight_no}]")
        for name, obj in to_load:
            state_dict = torch.load(
                os.path.join(dirname, f'{name}_{weight_no}.tar'), 
                map_location=device
            )
            if name == "weights":
                state_dict = {k2: state_dict[k1] for k1, k2 in match_names(state_dict, obj).items()}
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                obj.load_state_dict(new_state_dict)
            else:
                obj.load_state_dict(state_dict)
        epoch = weight_no
    else:
        epoch = 0

    masks = {}
    if masks_path and os.path.exists(masks_path):
        with open(masks_path, 'rb') as f:
            masks = pickle.load(f)
    
    return epoch, masks


class Trainer:
    def __init__(
        self, model, device, train_loader, valid_loader, criterion=None,
        use_amp=True, lr_scheduler_fn=None, restore_optim=False,
        save_optim_every=10, grad_accum_split=1, quantile_grad_clip=False,
        chunks_per_epoch=None, batch_size=None, masks=None
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion or model.loss
        self.use_amp = use_amp
        self.lr_scheduler_fn = lr_scheduler_fn or linear_warmup_cosine_decay()
        self.restore_optim = restore_optim
        self.save_optim_every = save_optim_every
        self.grad_accum_split = grad_accum_split
        self.scaler = torch.amp.GradScaler(enabled=use_amp)
        self.optimizer = None
        self.masks = masks or {}
        
        if quantile_grad_clip:
            self.clip_grad = ClipGrad()
        else:
            self.clip_grad = lambda parameters: torch.nn.utils.clip_grad_norm_(parameters, max_norm=2.0).item()

        self.batch_size = batch_size
        self.chunks_per_epoch = chunks_per_epoch
        self.steps_per_epoch = chunks_per_epoch // batch_size if chunks_per_epoch else None

        # Apply masks immediately after initialization
        if self.masks:
            self._apply_masks()

    def _apply_masks(self):
        """Apply masks to weights and zero out gradients in pruned positions"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    mask = self.masks[name].to(param.device)
                    
                    # Skip application if shapes don't match
                    if mask.shape != param.shape:
                        continue
                        
                    # Apply mask to weights
                    param.data.mul_(mask)
                    
                    # Zero out gradients in pruned positions
                    if param.grad is not None:
                        param.grad.data.mul_(mask)

    def train_one_step(self, batch):
        # Apply masks before forward pass
        if self.masks:
            self._apply_masks()
            
        self.optimizer.zero_grad()

        losses = None
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            for batch_ in zip(
                *map(lambda t: t.chunk(self.grad_accum_split, dim=0), batch)
            ):
                data_, targets_, lengths_, *args = (x.to(self.device) for x in batch_)

                scores_ = self.model(data_, *args)
                losses_ = self.criterion(scores_, targets_, lengths_)

                if not isinstance(losses_, dict): 
                    losses_ = {'loss': losses_}

                total_loss = losses_.get('total_loss', losses_['loss']) / self.grad_accum_split
                self.scaler.scale(total_loss).backward()

                losses = {
                    k: ((v.item() / self.grad_accum_split) if losses is None else (v.item() / self.grad_accum_split) + losses[k])
                    for k, v in losses_.items()
                }

        scale = self.scaler.get_scale()
        self.scaler.unscale_(self.optimizer)
        grad_norm = self.clip_grad(self.model.parameters())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Apply masks again after optimizer step to maintain sparsity
        if self.masks:
            self._apply_masks()

        return losses, grad_norm, scale

    def train_one_epoch(self, loss_log, lr_scheduler):
        t0 = perf_counter()
        chunks = 0
        self.model.train()

        progress_bar = tqdm(
            total=self.steps_per_epoch, desc=f'[0/{self.chunks_per_epoch}]',
            ascii=True, leave=True, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}{postfix}]',
            **tqdm_environ()
        )
        smoothed_loss = None

        with progress_bar:
            for batch in islice(self.train_loader, self.steps_per_epoch):
                chunks += batch[0].shape[0]
                losses, grad_norm, scale = self.train_one_step(batch)

                smoothed_loss = losses['loss'] if smoothed_loss is None else (0.01 * losses['loss'] + 0.99 * smoothed_loss)

                progress_bar.set_postfix(loss=f'{smoothed_loss:.4f}')
                progress_bar.set_description(f"[{chunks}/{self.chunks_per_epoch}]")
                progress_bar.update()

                if loss_log is not None:
                    lr = lr_scheduler.get_last_lr()
                    if len(lr) == 1: 
                        lr = lr[0]
                    loss_log.append({
                        'chunks': chunks,
                        'time': perf_counter() - t0,
                        'grad_norm': grad_norm,
                        'lr': lr,
                        'scale': scale,
                        **losses
                    })

                if lr_scheduler is not None: 
                    lr_scheduler.step()

        return smoothed_loss, perf_counter() - t0

    def validate_one_step(self, batch):
        data, targets, lengths, *args = batch
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            scores = self.model(data.to(self.device), *(x.to(self.device) for x in args))
            losses = self.criterion(scores, targets.to(self.device), lengths.to(self.device))
        losses = {k: v.item() for k, v in losses.items()} if isinstance(losses, dict) else losses.item()
        if hasattr(self.model, 'decode_batch'):
            seqs = self.model.decode_batch(scores)
        else:
            seqs = [self.model.decode(x) for x in permute(scores, 'TNC', 'NTC')]
        refs = [decode_ref(target, self.model.alphabet) for target in targets]

        n_pre = getattr(self.model, "n_pre_context_bases", 0)
        n_post = getattr(self.model, "n_post_context_bases", 0)
        if n_pre > 0 or n_post > 0:
            refs = [ref[n_pre:len(ref)-n_post] for ref in refs]

        accs = [
            accuracy(ref, seq, min_coverage=0.5) if len(seq) else 0. for ref, seq in zip(refs, seqs)
        ]
        return seqs, refs, accs, losses

    def validate_one_epoch(self):
        self.model.eval()
        with torch.no_grad():
            results = [self.validate_one_step(batch) for batch in self.valid_loader]
            seqs, refs, accs, losses = zip(*results)
        seqs, refs, accs = (sum(x, []) for x in (seqs, refs, accs))
        loss = np.mean([(x['loss'] if isinstance(x, dict) else x) for x in losses])
        return loss, np.mean(accs), np.median(accs)

    def init_optimizer(self, lr, **optim_kwargs):
        if "package" in optim_kwargs:
            optim_cls = getattr(
                import_module(optim_kwargs.pop('package')), 
                optim_kwargs.pop('symbol')
            )
        else:
            optim_cls = torch.optim.AdamW

        print(f"[loading optim] - '{optim_cls.__name__}' with args: {optim_kwargs}")
        optim_kwargs["lr"] = lr
        self.optimizer = optim_cls(self.model.parameters(), **optim_kwargs)

    def get_lr_scheduler(self, epochs, last_epoch=0):
        return self.lr_scheduler_fn(self.optimizer, self.steps_per_epoch, epochs, last_epoch)

    def fit(self, workdir, epochs=1, lr=2e-3, **optim_kwargs):
        if self.optimizer is None:
            self.init_optimizer(lr, **optim_kwargs)

        last_epoch, loaded_masks = load_state(
            workdir, 
            self.device, 
            self.model, 
            self.optimizer if self.restore_optim else None,
            masks_path=os.path.join(workdir, 'masks.pkl')
        )
        
        # Use loaded masks if we don't have any
        if not self.masks and loaded_masks:
            self.masks = loaded_masks
            # Reapply masks after loading state
            if self.masks:
                self._apply_masks()

        if self.restore_optim:
            for i, pg in enumerate(self.optimizer.param_groups):
                pg["initial_lr"] = pg["lr"] = lr[i] if isinstance(lr, (list, tuple)) else lr

        lr_scheduler = self.get_lr_scheduler(epochs, last_epoch=last_epoch)

        for epoch in range(1 + last_epoch, epochs + 1):
            try:
                with CSVLogger(os.path.join(workdir, f'losses_{epoch}.csv')) as loss_log:
                    train_loss, duration = self.train_one_epoch(loss_log, lr_scheduler)

                model_state = self.model.state_dict()
                torch.save(model_state, os.path.join(workdir, f"weights_{epoch}.tar"))
                
                # Save masks on first epoch and whenever they change
                if self.masks:
                    with open(os.path.join(workdir, 'masks.pkl'), 'wb') as f:
                        pickle.dump(self.masks, f)
                
                if epoch % self.save_optim_every == 0:
                    torch.save(self.optimizer.state_dict(), os.path.join(workdir, f"optim_{epoch}.tar"))

                val_loss, val_mean, val_median = self.validate_one_epoch()
            except KeyboardInterrupt:
                break

            print(f"[epoch {epoch}] directory={workdir} loss={val_loss:.4f} "
                  f"mean_acc={val_mean:.3f}% median_acc={val_median:.3f}%")

            with CSVLogger(os.path.join(workdir, 'training.csv')) as training_log:
                training_log.append({
                    'time': datetime.today(),
                    'duration': int(duration),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'validation_loss': val_loss,
                    'validation_mean': val_mean,
                    'validation_median': val_median
                })


def main(args):
    workdir = os.path.expanduser(args.training_directory)
    if os.path.exists(workdir) and not args.force:
        print(f"[error] {workdir} exists, use -f to force continue training.")
        sys.exit(1)
    os.makedirs(workdir, exist_ok=True)

    init(args.seed, args.device, (not args.nondeterministic))
    device = torch.device(args.device)

    if not args.pretrained:
        config = toml.load(args.config)
    else:
        dirname = args.pretrained
        if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models_dir__, dirname)):
            dirname = os.path.join(__models_dir__, dirname)
        pretrain_file = os.path.join(dirname, 'config.toml')
        config = toml.load(pretrain_file)
        if 'lr_scheduler' in config:
            print("[ignoring 'lr_scheduler' in --pretrained config]")
            del config['lr_scheduler']

    argsdict = dict(training=vars(args))
    argsdict["training"]["pwd"] = os.getcwd()

    print("[loading model]")
    masks = {}
    if args.pretrained:
        print(f"[using pretrained model {args.pretrained}]")
        model = load_model(args.pretrained, device, half=False)
        
        # Load masks if available
        masks_path = os.path.join(dirname, 'masks.pkl')
        if os.path.exists(masks_path):
            with open(masks_path, 'rb') as f:
                masks = pickle.load(f)
                print(f"Loaded pruning masks from {masks_path}")
    else:
        model = load_symbol(config, 'Model')(config)

    print("[loading data]")
    data = DataSettings(
        training_data=args.directory,
        num_train_chunks=args.chunks,
        num_valid_chunks=args.valid_chunks if args.valid_chunks is not None else args.chunks // 10,
        output_dir=workdir
    )
    model_setup = ModelSetup(
        n_pre_context_bases=getattr(model, "n_pre_context_bases", 0),
        n_post_context_bases=getattr(model, "n_post_context_bases", 0),
        standardisation=config.get("standardisation", {}),
    )
    compute_settings = ComputeSettings(
        batch_size=args.batch,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    train_loader, valid_loader = load_data(data, model_setup, compute_settings)

    try:
        dataset_cfg = train_loader.dataset.dataset_config
    except AttributeError:
        dataset_cfg = {}
    toml.dump({**config, **argsdict, **dataset_cfg}, open(os.path.join(workdir, 'config.toml'), 'w'))

    if config.get("lr_scheduler"):
        sched_config = config["lr_scheduler"]
        lr_scheduler_fn = getattr(
            import_module(sched_config["package"]), sched_config["symbol"]
        )(**sched_config)
    else:
        lr_scheduler_fn = None

    trainer = Trainer(
        model, device, train_loader, valid_loader,
        use_amp=not args.no_amp,
        lr_scheduler_fn=lr_scheduler_fn,
        restore_optim=args.restore_optim,
        save_optim_every=args.save_optim_every,
        grad_accum_split=args.grad_accum_split,
        quantile_grad_clip=args.quantile_grad_clip,
        chunks_per_epoch=args.chunks,
        batch_size=args.batch,
        masks=masks
    )

    if ',' in args.lr:
        lr = [float(x) for x in args.lr.split(',')]
    else:
        lr = float(args.lr)
    
    optim_kwargs = config.get("optim", {})
    trainer.fit(workdir, args.epochs, lr, **optim_kwargs)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("training_directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--config', default=default_config)
    group.add_argument('--pretrained', default="")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", default='2e-3')
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--chunks", default=0, type=int)
    parser.add_argument("--valid-chunks", default=None, type=int)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    parser.add_argument("--restore-optim", action="store_true", default=False)
    parser.add_argument("--nondeterministic", action="store_true", default=False)
    parser.add_argument("--save-optim-every", default=10, type=int)
    parser.add_argument("--grad-accum-split", default=1, type=int)
    quantile_group = parser.add_mutually_exclusive_group()
    quantile_group.add_argument('--quantile-grad-clip', dest='quantile_grad_clip', action='store_true')
    quantile_group.add_argument('--no-quantile-grad-clip', dest='quantile_grad_clip', action='store_false')
    quantile_group.set_defaults(quantile_grad_clip=True)
    parser.add_argument("--num-workers", default=4, type=int)
    return parser


if __name__ == "__main__":
    args = argparser().parse_args()
    main(args)