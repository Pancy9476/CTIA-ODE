import os
import torch
from data_provider.data_loader import load_and_split_data
from models.CTIA_ODE import CTIAODEForecaster
from utils.tools import log_to_file, evaluate_model
from utils.losses import ORHL


def train_and_evaluate(args, device, log_file_path):
    csv_path = os.path.join(args.root_path, args.data_path)
    (
        train_loader,
        val_loader,
        test_loader,
        num_vars,
        num_time_features
    ) = load_and_split_data(
        csv_path=csv_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size
    )

    model = CTIAODEForecaster(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_vars=num_vars,
        num_time_features=num_time_features,
        d_model=args.d_model,
        primary_idx=-1,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    criterion = ORHL(delta=1.0, lambda_=5.0)

    best_val_mse = float("inf")
    patience_counter = 0

    # Ensure weights directory exists
    weight_dir = os.path.join("./checkpoints", args.dataset_name)
    os.makedirs(weight_dir, exist_ok=True)
    best_model_path = os.path.join(weight_dir, f"best_model_T{args.pred_len}.pth")

    log_to_file(log_file_path, (
        f"\n{'=' * 78}\n"
        f"M -> M CTIA-ODE Forecasting\n"
        f"Dataset    : {args.dataset_name}\n"
        f"Seq Len    : {args.seq_len}\n"
        f"Pred Len   : {args.pred_len}\n"
        f"Variables  : {num_vars}\n"
        f"D Model    : {args.d_model}\n"
        f"Batch Size : {args.batch_size}\n"
        f"LR         : {args.learning_rate}\n"
        f"Loss       : ORHL\n"
        f"{'=' * 78}"
    ))

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for x, history_time, future_time, y in train_loader:
            x = x.to(device, non_blocking=True)
            history_time = history_time.to(device, non_blocking=True)
            future_time = future_time.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds = model(x, history_time, future_time)

            loss = criterion(preds, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        train_loss = total_train_loss / max(num_batches, 1)

        val_mse, val_mae = evaluate_model(model, val_loader, device)
        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            status = "BEST"
        else:
            patience_counter += 1
            status = ""

        epoch_msg = (
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Val MAE: {val_mae:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"ES: {patience_counter}/{args.patience} {status}"
        )
        log_to_file(log_file_path, epoch_msg)

        if patience_counter >= args.patience:
            log_to_file(log_file_path, "Early stopping triggered.")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_mse, test_mae = evaluate_model(model, test_loader, device)

    final_msg = f"\nFINAL RESULT | T={args.pred_len} | MSE={test_mse:.6f} | MAE={test_mae:.6f}"
    log_to_file(log_file_path, final_msg)

    return test_mse, test_mae