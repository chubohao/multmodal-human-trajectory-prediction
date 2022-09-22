from torch.utils.data import DataLoader

from data.trajectories_full import TrajectoryDataset, Collate


def data_loader(args, path, augment = False):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)
    collate = Collate(augment)
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        collate_fn=collate.seq_collate,
        pin_memory=True)
    return dset, loader
