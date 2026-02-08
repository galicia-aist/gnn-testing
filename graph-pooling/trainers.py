import torch
from utils import eval_metrics
from sklearn.metrics import roc_auc_score

def train(model, optimizer, loss_fn, data, node_train_ids, y_train,  loader=None, device=None, logger=None):
    if loader is None:
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[node_train_ids], y_train)
        loss.backward()
        optimizer.step()
        acc_train = eval_metrics(out, node_train_ids, y_train)
        loss_train = loss.item()
    else:
        loss_train, acc_train, _ = process_with_loader(loader, model, node_train_ids, y_train, loss_fn, device,
                                                       mode="train", optimizer=optimizer, compute_auc=False,
                                                       logger=None)
    return loss_train, acc_train

@torch.no_grad()
def evaluate(model, loss_fn, data, node_eval_ids, y_eval, loader=None, device=None, logger=None):
    if loader is None:
        model.eval()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[node_eval_ids], y_eval)
        acc_eval = eval_metrics(out, node_eval_ids, y_eval)
        auc_eval = roc_auc_score(y_eval.cpu().numpy(), out[node_eval_ids].cpu().numpy())
        loss_eval = loss.item()
    else:
        loss_eval, acc_eval, auc_eval = process_with_loader(loader, model, node_eval_ids, y_eval, loss_fn, device,
                                                       mode="eval", optimizer=None, compute_auc=True,
                                                       logger=None)
    return loss_eval, acc_eval, auc_eval

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
