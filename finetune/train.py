"""Fine-tune Chronos-Bolt-small on Bitcoin. The test period is cut off before training starts.

Tries a few settings and keeps the one with the lowest error on the validation period.
"""

import argparse
import json
import time

import numpy as np
import torch
from chronos import BaseChronosPipeline

from common import (BASE_MODEL, EVALUATION_FILE, MODEL_DIR, TRAINING_FILE, device, fc,
                    forecast_median, gap_free_starts, load_prices, when, windows)

TEST_HOURS = 365 * 24
VAL_HOURS = 182 * 24
# (learning rate, how many hours ahead the model practises on)
SETTINGS = [(1e-5, 6), (1e-5, 64), (5e-5, 6)]


def train_one(settings, hours, prices, val_start, val_x, val_y, args, dev):
    """Returns (best validation error, trained model)."""
    lr, target = settings
    pipe = BaseChronosPipeline.from_pretrained(BASE_MODEL, device_map=dev, dtype=torch.float32)
    net = pipe.model
    size = fc.CONTEXT_HOURS + target
    starts = gap_free_starts(hours, size)
    starts = starts[hours[starts + size - 1] < val_start]  # windows must end before validation begins
    offsets = np.arange(size)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    use_fp16 = args.precision == "fp16" and dev == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_fp16)

    def validate():
        net.eval()
        error = float((np.abs(forecast_median(pipe, val_x) - val_y) / val_y).mean() * 100)
        net.train()
        return error

    print(f"\nlr={lr} target={target}h, {len(starts):,} training windows. "
          f"Original model's validation error: {validate():.4f}%", flush=True)
    best, best_state, bad = float("inf"), None, 0
    for step in range(1, args.steps + 1):
        for group in opt.param_groups:  # short warm-up, then decay to zero
            group["lr"] = lr * min(step / 100, 1 - step / (args.steps + 1))
        batch = prices[rng.choice(starts, args.batch)[:, None] + offsets]
        x = torch.tensor(batch[:, :fc.CONTEXT_HOURS], device=dev)
        y = torch.tensor(batch[:, fc.CONTEXT_HOURS:], device=dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=use_fp16):
            loss = net(context=x, target=y).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss became {loss.item()} at step {step} (try --precision fp32)")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % args.eval_every == 0 or step == args.steps:
            error = validate()
            print(f"  step {step:>5}  loss {loss.item():.4f}  val error {error:.4f}%", flush=True)
            if error < best:
                best, bad = error, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    print("  no more improvement, stopping early")
                    break
    net.load_state_dict(best_state)
    net.eval()
    return best, pipe


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--quick", action="store_true", help="tiny run to check that everything works; saves nothing")
    args = p.parse_args()
    if args.quick:
        args.steps, args.eval_every = 4, 2

    dev = device()
    print(f"Training on {torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU'} ({args.precision})")
    hours, prices = load_prices()
    test_start = int(hours[-1]) - TEST_HOURS + 1
    val_start = test_start - VAL_HOURS
    # Cut the test period off right here: nothing below can see it.
    keep = hours < test_start
    hours, prices = hours[keep], prices[keep]

    starts = gap_free_starts(hours)
    val = starts[hours[starts + fc.CONTEXT_HOURS] >= val_start]
    val_x, val_y = windows(prices, val[::3])
    print(f"Train: {when(hours[0])} to {when(val_start - 1)}\n"
          f"Val:   {when(val_start)} to {when(test_start - 1)} ({len(val_x)} forecasts)\n"
          f"Test:  {when(test_start)} onwards (hidden)")

    results, best_error, best_pipe, best_settings = [], float("inf"), None, None
    t0 = time.time()
    for settings in SETTINGS[:1] if args.quick else SETTINGS:
        error, pipe = train_one(settings, hours, prices, val_start, val_x, val_y, args, dev)
        results.append({"lr": settings[0], "target_hours": settings[1], "val_error_percent": error})
        if error < best_error:
            best_error, best_settings = error, settings
            best_pipe = pipe
            pipe.model.to("cpu")  # keep GPU memory free for the next run
        if dev == "cuda":
            print(f"  peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    if args.quick:
        print("Quick check OK, nothing saved.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_FILE.unlink(missing_ok=True)  # a new model has not earned its place yet
    best_pipe.model.save_pretrained(MODEL_DIR)
    TRAINING_FILE.write_text(json.dumps({
        "base_model": BASE_MODEL, "train_start": int(hours[0]), "val_start": val_start, "test_start": test_start,
        "lr": best_settings[0], "target_hours": best_settings[1], "val_error_percent": best_error,
        "precision": args.precision, "device": dev, "all_runs": results,
        "minutes": round((time.time() - t0) / 60, 1),
    }, indent=2))
    print(f"\nBest: lr={best_settings[0]} target={best_settings[1]}h, val error {best_error:.4f}%. Saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
