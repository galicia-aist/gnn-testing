import torch
from utils import eval_metrics
from sklearn.metrics import roc_auc_score

def train(model, optimizer, loss_fn, data, node_train_ids, y_train):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[node_train_ids], y_train)
    loss.backward()
    optimizer.step()
    return loss.item(), out

@torch.no_grad()
def test(model, out, node_train_ids, node_val_ids, node_test_ids, y_train, y_val, y_test, loss_fn):
    model.eval()

    # Accuracies
    acc_train = eval_metrics(out, node_train_ids, y_train)
    acc_val = eval_metrics(out, node_val_ids, y_val)
    acc_test = eval_metrics(out, node_test_ids, y_test)

    # Validation AUC
    auc_val = roc_auc_score(y_val.cpu().numpy(), out[node_val_ids].cpu().numpy())

    # Validation loss
    loss_val = loss_fn(out[node_val_ids], y_val).item()

    return acc_train, acc_val, acc_test, loss_val, auc_val

def process_with_loader(
    loader,
    model,
    node_ids,
    y_true,
    loss_fn,
    device,
    mode="train",
    optimizer=None,
    compute_auc=False,
    logger=None,
):
    is_train = mode == "train"
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_logits = []
    all_labels = []

    if is_train:
        model.train()
    else:
        model.eval()

    for batch in loader:
        batch = batch.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            out = model(batch.x, batch.edge_index).view(-1)

            # -------------------------------------------------
            # FIX: global → local index mapping
            # -------------------------------------------------
            # batch.n_id: global node ids
            # out: predictions for local batch nodes
            global_to_local = torch.full(
                (batch.n_id.max() + 1,),
                -1,
                dtype=torch.long,
                device=batch.n_id.device,
            )
            global_to_local[batch.n_id] = torch.arange(
                batch.n_id.size(0), device=batch.n_id.device
            )

            local_idx = global_to_local[node_ids]
            valid = local_idx >= 0

            if valid.sum() == 0:
                continue

            out_sel = out[local_idx[valid]]
            y_sel = y_true[valid]

            loss = loss_fn(out_sel, y_sel)

            if is_train:
                loss.backward()
                optimizer.step()

        # ---- Metrics (shared) ----
        preds = (out_sel > 0).long()
        correct = (preds == y_sel.long()).sum().item()

        batch_size = y_sel.size(0)
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_size

        if compute_auc:
            all_logits.append(out_sel.detach().cpu())
            all_labels.append(y_sel.detach().cpu())

    if total_samples == 0:
        return 0.0, 0.0, None

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    auc = None
    if compute_auc and all_logits:
        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        auc = roc_auc_score(labels, logits)

    return avg_loss, avg_acc, auc
