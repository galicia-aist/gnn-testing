# --- Standard library imports ---
import sys
import math
import time

# --- Third-party libraries ---
from sklearn.metrics import roc_auc_score

# --- Local project modules ---
from models import GCN, GAT, GraphSAGE
from utils import *
from bag_creation import get_eval_nodes
from trainers import train, evaluate, process_with_loader
# from shared.utils import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_experiment(chosen_dataset, chosen_model, device, bag_ratio, single_layer, val_interval, test_interval, hidden_units=64,
                   dropout_rate=0.5, lr=0.01, weight_decay=5e-4, max_epochs=500, logger=None):

    dataset = load_data(chosen_dataset)

    # Pick a chosen class for binary classification

    if chosen_dataset.lower() in {"arxiv", "products"}:
        labels = encode_onehot(dataset.y.squeeze().numpy())
    else:
        labels = encode_onehot(dataset.y.numpy())

    labels = torch.LongTensor(np.where(labels)[1])

    num_class = labels.max().item() + 1
    chosen_label = random.randint(0, num_class - 1)

    label_node, label_non_node = split_labels(labels, num_class)

    m = 500  # number of bag (node-level tasks)
    n = 2  # number of instances within a bag (node-level standard MIL)

    # Pick a chosen class and consistent evaluation nodes

    idx_train, idx_val, idx_test, node_train, node_eva = get_eval_nodes(m, n, chosen_label, dataset.y, label_node, label_non_node, bag_ratio, logger=logger)

    node_train_keys = list(node_train.keys())
    node_eva_keys = list(node_eva.keys())

    node_train_ids = torch.tensor([node_train_keys[i][0] for i in idx_train], dtype=torch.long, device=device)
    y_train = torch.tensor([node_train[node_train_keys[i]] for i in idx_train], dtype=torch.float, device=device)

    # split node_eva into val/test
    node_val_ids = torch.tensor( [node_eva_keys[i][0] for i in idx_val], dtype=torch.long, device=device)
    y_val = torch.tensor([node_eva[node_eva_keys[i]] for i in idx_val], dtype=torch.float, device=device)
    node_test_ids = torch.tensor([node_eva_keys[i][0] for i in idx_test], dtype=torch.long, device=device)
    y_test = torch.tensor([node_eva[node_eva_keys[i]] for i in idx_test], dtype=torch.float, device=device)

    if chosen_dataset.lower() in {"reddit", "products"}:
        train_mask = make_input_nodes(node_train_ids.cpu(), dataset.num_nodes)
        val_mask = make_input_nodes(node_val_ids.cpu(), dataset.num_nodes)
        test_mask = make_input_nodes(node_test_ids.cpu(), dataset.num_nodes)

        dataset.train_mask = train_mask
        dataset.val_mask = val_mask
        dataset.test_mask = test_mask

        train_loader, val_loader, test_loader = get_loaders(
            args, dataset, logger=logger
        )
    else:
        train_loader = val_loader = test_loader = None

    # ---- GCN training same as before ----
    data = dataset.to(device)
    if chosen_model.lower() == "gcn":
        model = GCN(dataset.num_features, hidden_units, dropout_rate, single_layer=single_layer).to(device)
    elif chosen_model.lower() == "gat":
        model = GAT(dataset.num_features, hidden_units, dropout_rate, single_layer=single_layer, heads=1).to(device)  # adjust heads if needed
    elif chosen_model.lower() == "sage":
        model = GraphSAGE(dataset.num_features, hidden_units, dropout_rate, single_layer=single_layer).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # ---- Training loop ----
    test_results = []
    start_time = time.time()
    best_val_acc = 0
    best_test_acc = 0

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # ---- Training ----
        loss_train, acc_train = train(model, optimizer, loss_fn, data, node_train_ids, y_train, loader=train_loader,
                                      device=device)

        # ---- Validation ----
        if epoch % val_interval == 0:
            loss_val, acc_val, auc_val = evaluate(model, loss_fn, data, node_val_ids, y_val, loader=val_loader,
                                      device=device)


        epoch_time = time.time() - epoch_start
        reach_time = time.time() - start_time

        # Update best
        if acc_val > best_val_acc:
            best_val_acc = acc_val

        # Per-epoch print
        logger.info(
            f"Epoch: {epoch:04d} | loss_train: {loss_train:.4f} | acc_train: {acc_train:.4f} | "
            f"loss_val: {loss_val:.4f} | acc_val: {acc_val:.4f} | auc_val: {auc_val:.4f} | "
            f"time_taken: {epoch_time:.4f}s | reach_time: {reach_time:.4f}s"
        )

        # ---- Test ----
        if epoch % test_interval == 0:
            loss_test, acc_test, auc_test = evaluate(model, loss_fn, data, node_test_ids, y_test, loader=test_loader,
                                      device=device)

            test_time = time.time() - start_time

            logger.info(
                f"[Test evaluation at epoch {epoch}] loss= {loss_test:.4f} | "
                f"accuracy= {acc_test:.4f} | auc= {auc_test:.4f}"
            )

            # Store test result
            test_results.append({
                "epoch": epoch,
                "loss": loss_test,
                "accuracy": acc_test,
                "auc": auc_test,
                "time": test_time
            })

    # ---- Final evaluation ----
    total_time = time.time() - start_time

    loss_test, acc_test, auc_test = evaluate(model, loss_fn, data, node_test_ids, y_test, loader=test_loader,
                                      device=device)

    logger.info(
        f"Final Test set results: loss= {loss_test:.4f} | accuracy= {acc_test:.4f} | "
        f"auc= {auc_test:.4f} | total time= {total_time:.2f}s"
    )

    # Write experiment summary
    filename = write_experiment_summary(args, test_results, total_time)

    if logger:
        logger.info(f"Results written to {filename}")

    return test_results, filename

if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    logger_settings = {
        "logger": {
            "model": args.model,
            "log_path": args.log_path,
            "dataset": args.d,
            "log_level": args.log_level.upper()
        },
        "ddp": args.ddp
    }

    with open("global_settings.json", "w") as file:
        json.dump(logger_settings, file, indent=4)

    logger = get_logger()

    log_experiment_settings(logger, args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    run_experiment(
        args.d,
        args.model,
        device,
        args.bag_ratio,
        args.single_layer,
        args.val_interval,
        args.test_interval,
        hidden_units=args.hidden,
        dropout_rate=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        logger=logger
    )