import argparse
import random
import torch.multiprocessing as mp
import torch
import numpy as np
from torch_geometric.datasets import Planetoid, CitationFull, Reddit
from utils import *
from torch_geometric.loader import NeighborLoader
from torch.utils.data.distributed import DistributedSampler
from sgc import SGC, train, evaluate
from gcn import GCN
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from data_loader import get_training_data


def main(rank, world_size, args, device, logger=None):
    batch_size_train = 1024
    batch_size_test = 2048

    logger.debug("before loading data")

    data = get_training_data(args.dataset)

    logger.debug("after loading data")

    if args.dataset == "Reddit":
        if args.ddp:
            train_sampler = DistributedSampler(
                data.train_mask.nonzero().squeeze(),  # Get only training indices
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )

            test_sampler = DistributedSampler(
                data.test_mask.nonzero().squeeze(),  # Get only test indices
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )

            train_loader = NeighborLoader(
                data,
                num_neighbors=[10, 10],  # Sample 10 neighbors per layer
                batch_size=batch_size_train,
                input_nodes=data.train_mask,
                sampler=train_sampler  # Ensure each GPU only gets a part of the dataset
            )

            test_loader = NeighborLoader(
                data,
                num_neighbors=[10, 10],
                batch_size=batch_size_test,
                input_nodes=data.test_mask,
                sampler=test_sampler  # Ensure correct test distribution
            )
        else:
            train_loader = NeighborLoader(
                data,
                num_neighbors=[10, 10],  # Sample 10 neighbors per layer
                batch_size=batch_size_train,
                input_nodes=data.train_mask
            )

            test_loader = NeighborLoader(
                data,
                num_neighbors=[10, 10],
                batch_size=batch_size_test,
                input_nodes=data.test_mask
            )
    else:
        train_loader = None
        test_loader = None
    if args.model == "GCN":
        model = GCN(data, args.hidden).to(device)
    else:
        model = SGC(data).to(device)
    if args.ddp:
        model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs):
        loss = train(model, optimizer, device, data, train_loader=train_loader, dataset_name=args.dataset)
        logger.debug(f'Epoch {epoch}: Loss: {loss:.4f}')

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

