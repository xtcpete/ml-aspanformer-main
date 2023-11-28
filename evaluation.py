import argparse
from src.datasets.data_module import DataModule
import torch
from src.ASpanFormer.aspanformer import ASpanFormer
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config, move_to_device
from src.utils.metrics import *
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
import json
from tqdm import tqdm

# from QuadtreeFeatureMatching.src.loftr import LoFTR as LoFTR_quad, default_cfg as default_cfg_quad


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # data_path for training data
    parser.add_argument("data_path", type=str, default="data", help="data path")

    # batch size per gpu
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size per gpu")

    # number of CPU cores to use
    parser.add_argument("--num_workers", type=int, default=4)

    # false to run on cpu
    parser.add_argument(
        "--force_cpu", action="store_true", help="Force pytorch to run in CPU mode.", default=False
    )

    parser.add_argument(
        "--weight_path", type=str, default=None, help='/path/to/weights'
    )

    parser.add_argument(
        "--Visual", action='store_true', default=False, help="Use Adamatcher")
    
    parser.add_argument(
        "--Ours", action='store_true', default=False, help="using our method")

    return parser.parse_args()


def main():
    args = parse_args()

    config = get_cfg_defaults()
    config.BATCH_SIZE = args.batch_size
    config.DATA_PATH = args.data_path
    config.NUM_WORKERS = args.num_workers
    config.DEVICE = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
    config = lower_config(config)
    weight_path = args.weight_path
    device = config['device']

    if args.Ours:
        config['aspan']['match_coarse']['iter'] = 10
        config['aspan']['match_coarse']['thr'] = 0.2
    matcher = ASpanFormer(lower_config(config['aspan']))
    weights = torch.load(weight_path, map_location="cpu")["state_dict"]
    # remove the first 'module.' in the state_dict
    weights = {k.replace('matcher.', ''): v for k, v in weights.items()}
    matcher.load_state_dict(weights)
    matcher.to(device)
    matcher.eval()
    print("Load weights successfully!")

    data_module = DataModule(args, config['aspan'])
    val_dataloader = data_module.validation_dataloader(subset=False)
    print("Initialize validation dataloader")

    R_errs = []
    t_errs = []
    epi_errs_list = []
    precision = []
    num_matches = []
    tb = len(val_dataloader)

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_dataloader)):
            move_to_device(batch, device)
            
            matcher(batch)

            b_mask = batch['m_bids'] == 0
            img0 = (batch['image0'][0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.int32)

            img1 = (batch['image1'][0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.int32)
            kpts0 = batch['mkpts0_f'][b_mask].cpu().numpy()
            kpts1 = batch['mkpts1_f'][b_mask].cpu().numpy()
            mconf = batch['mconf'][b_mask].cpu().numpy()
            index = np.where(mconf >= 0.2)[0]

            if 'scale0' in batch:
                kpts0 = kpts0 / batch['scale0'][0].cpu().numpy()[[1, 0]]
                kpts1 = kpts1 / batch['scale1'][0].cpu().numpy()[[1, 0]]
            compute_pose_errors(batch)
            
            compute_symmetrical_epipolar_errors(batch)
            epi_errs_list.extend(batch['epi_errs'])
            
            text = [
                'R_errs: {}'.format(batch['R_errs'][0])
            ]
            if args.Visual:
                color = cm.jet_r(mconf, alpha=0.7)
                make_matching_figure(img0, img1, kpts0[index], kpts1[index], text=text, color=color[index], path=f'./src/logs/images/{batch_idx}.png')
            R_errs.extend(batch['R_errs'])
            t_errs.extend(batch['t_errs'])
            epi_errs = batch['epi_errs'].detach().cpu()[b_mask.cpu().numpy()]
            correct_mask = epi_errs < 1e-4
            correct_mask = correct_mask.cpu().numpy()
            pre = np.mean(correct_mask) if len(correct_mask) > 0 else 0
            num_matches.extend([len(index)])
            precision.extend([pre])

    metrics = {'R_errs': R_errs, 'precision': precision, 'num_matches': num_matches}
    if args.Ours:
        type = 'Ours'
    else:
        type = 'Aspan'
    
    with open(f'./src/logs/{type}_metrics.json', 'w') as f:
        json.dump(metrics, f)
    
if __name__ == "__main__":
    main()