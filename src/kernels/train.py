from collections.abc import Callable

import torch
import tqdm
from jaxtyping import Float, Integer


def train(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    lr_sched: torch.optim.lr_scheduler.LRScheduler,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    loss_fn: Callable[[Float[torch.Tensor, "..."], Integer[torch.Tensor, "..."]], Float[torch.Tensor, " batch"]],
    epochs: int,
    max_norm: float | None = None,
) -> dict[str, list]:
    """Train a model with optional grad clipping and periodic validation.

    Args:
        model: Module to train.
        opt: Optimizer used for parameter updates.
        lr_sched: Learning-rate scheduler stepped each iteration.
        train_dl: DataLoader yielding (x, y) training batches.
        valid_dl: DataLoader used to compute per-epoch validation losses.
        loss_fn: Callable producing per-sequence loss from model outputs and targets.
        epochs: Number of epochs to iterate over ``train_dl``/``valid_dl``.
        max_norm: Optional gradient norm clip threshold.

    Returns:
        dict[str, list]: History containing ``losses`` (per train step), ``lrs`` (learning rate per train step),
        ``valid_epoch_losses`` (per validation epoch), and ``norms`` (per train step) if ``max_norm`` was set.
    """
    losses = []
    lrs = []
    norms = []

    valid_epoch_losses = []

    for epoch in range(epochs):
        # train
        model.train()
        with tqdm.tqdm(desc=f"epoch {epoch + 1}/{epochs}, train", total=len(train_dl)) as pbar:
            for i, (x, y) in enumerate(train_dl):
                model.zero_grad(set_to_none=True)

                x = x.to("cuda")
                y = y.to("cuda")

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(x)

                loss = loss_fn(out.flatten(end_dim=1), y.flatten()).mean()

                loss.backward()

                if max_norm is not None:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    norms.append(norm.item())

                opt.step()
                lr_sched.step()

                losses.append(loss.item())
                lrs.append(lr_sched.get_last_lr())

                pbar.set_postfix_str(f"loss={losses[-1]:5.3f}")
                pbar.update(1)

        # validate
        model.eval()
        with tqdm.tqdm(desc=f"epoch {epoch + 1}/{epochs}, valid", total=len(valid_dl)) as pbar:
            valid_losses = []
            for i, (x, y) in enumerate(valid_dl):
                x = x.to("cuda")
                y = y.to("cuda")

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(x)

                loss = loss_fn(out.flatten(end_dim=1), y.flatten()).mean()
                valid_losses.append(loss.item())

                pbar.set_postfix_str(f"valid_loss={valid_losses[-1]:5.3f}")
                pbar.update(1)

            valid_epoch_losses.append(sum(valid_losses) / len(valid_losses))
            pbar.set_postfix_str(f"valid_loss={valid_epoch_losses[-1]:5.3f}")

    out = {"lrs": lrs, "losses": losses, "valid_epoch_losses": valid_epoch_losses}
    if max_norm is not None:
        out["norms"] = norms
    return out
