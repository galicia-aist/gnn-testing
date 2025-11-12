from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import NeighborLoader


def get_loaders(args, data, world_size=1, rank=0, logger=None):
    """
    Creates NeighborLoaders with or without DistributedSamplers depending on args.ddp.

    Args:
        args: parsed command-line arguments (expects args.ddp and args.d)
        data: PyG dataset object
        idx_train, idx_val, idx_test: node indices for each split
        world_size: number of processes (GPUs)
        rank: current process rank

    Returns:
        train_loader, val_loader, test_loader
    """

    if args.loaders:
        def make_loader(mask, neighbors, shuffle, sampler=None):
            return NeighborLoader(
                data,
                num_neighbors=neighbors,
                input_nodes=mask,
                batch_size=1024,
                shuffle=shuffle,
                sampler=sampler
            )

        if getattr(args, "ddp", False):
            # --- Distributed samplers ---
            def make_sampler(mask, shuffle):
                return DistributedSampler(
                    mask.nonzero().squeeze(),
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle
                )

            train_loader = make_loader(data.train_mask, [10, 5], True, make_sampler(data.train_mask, True))
            val_loader = make_loader(data.val_mask, [25, 10], True, make_sampler(data.val_mask, True))
            test_loader = make_loader(data.test_mask, [25, 10], False, make_sampler(data.test_mask, False))

        else:
            # --- Regular loaders ---
            train_loader = make_loader(data.train_mask, [5, 4], True)
            val_loader = make_loader(data.val_mask, [25, 10], False)
            test_loader = make_loader(data.test_mask, [25, 10], False)

        logger.debug(f"Train loader batches: {len(train_loader)}")
        logger.debug(f"Validation loader batches: {len(val_loader)}")
        logger.debug(f"Test loader batches: {len(test_loader)}")

        return train_loader, val_loader, test_loader

    return None, None, None