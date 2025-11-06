# === Standard Library ===
import random
import time

# === Third-Party Libraries ===
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# === Local Modules ===
from data_loader import get_training_data
from gcn import GCN
from sgc import SGC, train, evaluate
from utils import *
from shared.utils import *


def main(rank, world_size, args, device, logger=None):

    logger.debug("before loading data")

    data = get_training_data(args.dataset)

    logger.debug("after loading data")

    train_loader, eval_loader, test_loader = get_loaders(args, data, world_size=world_size, rank=rank, logger=logger)

    if args.model == "GCN":
        model = GCN(data, args.hidden).to(device)
    elif args.model == "GCN":
        model = SGC(data).to(device)
    else:
        quit("No model selected")


    if args.ddp:
        model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs):
        start_time = time.time()

        loss = train(model, optimizer, device, data, train_loader=train_loader, dataset_name=args.dataset)

        elapsed = time.time() - start_time
        logger.info(f"Epoch {epoch}: Loss: {loss:.4f} | Time: {elapsed:.2f}s")

    if args.ddp:
        dist.barrier()
        if rank == 0:
            accuracy = evaluate(model, device, data, test_loader=test_loader, dataset_name=args.dataset)
            logger.info(f"Test Accuracy: {accuracy:.4f}")
    else:
        accuracy = evaluate(model, device, data, test_loader=test_loader, dataset_name=args.dataset)
        logger.info(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    args = get_args()

    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    logger_settings = {
        "logger": {
            "model": args.model,
            "log_path": args.log_path,
            "dataset": args.dataset,
            "log_level": args.log_level.upper()
        },
        "ddp": args.ddp
    }

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open("global_settings.json", "w") as file:
        json.dump(logger_settings, file, indent=4)

    logger = get_logger()

    log_experiment_settings(logger, args)

    world_size = None

    if args.ddp:
        world_size = torch.cuda.device_count()
        mp.set_start_method("spawn", force=True)
        mp.spawn(main, args=(world_size, args, device), nprocs=world_size, join=True)
    else:
        main(0, 0, args, device, logger=logger)

