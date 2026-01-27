import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from dataloaders.dataset_embed_multiple_prior_img import  BreastCancerRiskDataset_multiple_Prior_EMBED
from dataloaders.dataset_csaw import BreastCancerRiskDatasetCSAWCC
from dataloaders.dataset_embed import BreastCancerRiskDataset
from evaluate.test_risk_prediction import test_risk
from dataloaders.dataset_embed_CC_or_MLO import BreastCancerRiskDataset_One_View
from dataloaders.dataset_csaw_CC_or_MLO import BreastCancerRiskDatasetCSAWCC_oneView

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--path_out_dir", type=str, required=True)
    parser.add_argument("--id_training", type=int, required=True)
    parser.add_argument("--path_test_folder", type=str, required=True)
    parser.add_argument("--num_epoch", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--use_only_multiview_pred", action="store_true", help="use only multi view features for prediction")
    parser.add_argument("--use_no_attention", action="store_true", help="ablation study: remove Attention")
    parser.add_argument("--use_only_one_view", type=str)
    parser.add_argument("--use_no_prior_img", action="store_true", help="ablation study: No prior images")
    parser.add_argument("--use_multiple_prior_img", action="store_true", help="Using multiple prior images")
    parser.add_argument("--perceiver_depth", default=2, type=int)
    parser.add_argument("--latent_dim", default=256, type=int)
    parser.add_argument("--self_per_cross_attn", default=2, type=int)
    parser.add_argument("--dropout_perceiver", default=0.0, type=float)
    parser.add_argument("--latent_dim_head", default=1, type=int)
    parser.add_argument("--num_latents", default=16, type=int)
    # Model architecture flags
    parser.add_argument("--early_stop", type=str, default="False")
    parser.add_argument("--best_model", type=str, default="False")
    parser.add_argument("--use_checkppoint", type=str, default="False")

    # Dataloader args
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--shuffle", default=False, type=bool)  # Corrected spelling
    parser.add_argument("--pin_memory", default=True, type=bool)
    parser.add_argument("--seed", default=2023, type=int)

    return parser.parse_args()


def custom_collate(batch):
    """
    Collate function to handle variable-length prior images.
    Keeps lists of prior images and time gaps as lists instead of stacking them.
    """
    batch_dict = {}

    for key in batch[0]:
        values = [sample[key] for sample in batch]

        # Keep variable-length lists as-is
        if key in ['previous_image_cc', 'previous_image_mlo', 'time_gap']:
            batch_dict[key] = values  # list of lists
        else:
            # Only stack if the elements are tensors
            if isinstance(values[0], torch.Tensor):
                batch_dict[key] = torch.stack(values, dim=0)
            else:
                batch_dict[key] = values  # fallback for non-tensors
    return batch_dict



def main():
    args = parse_arguments()
    accelerator = Accelerator()

    # Set seed for reproducibility on all processes
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.use_only_one_view == "True":
        print("use either CC or MLO")
        if args.dataset == "CSAW":
            print("Use CSAW-CC dataset")
            test_dataset = BreastCancerRiskDatasetCSAWCC_oneView(
                args.csv_file, args.data_root, "test", transforms=None
            )
        else:
            print("Use EMBED dataset")
            test_dataset = BreastCancerRiskDataset_One_View(
                args.csv_file, args.data_root, "test", transforms=None
            )
    elif args.use_multiple_prior_img:
        print("use multiple prior images")
        if args.dataset == "CSAW":
            print("Use CSAW-CC dataset")
            test_dataset = BreastCancerRiskDatasetCSAWCC(
                args.csv_file, args.data_root, "test", transforms=None
            )

        else:
            print("Use EMBED dataset")
            test_dataset = BreastCancerRiskDataset_multiple_Prior_EMBED(
                args.csv_file, args.data_root, "test", transforms=None
            )
    else:
        if args.dataset == "CSAW":
            print("Use CSAW-CC dataset")
            test_dataset = BreastCancerRiskDatasetCSAWCC(
                args.csv_file, args.data_root, "test", transforms=None
            )
        else:
            print("Use EMBED dataset")
            test_dataset  = BreastCancerRiskDataset(
                args.csv_file, args.data_root, "test", transforms=None
            )
    if args.use_multiple_prior_img:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            pin_memory=args.pin_memory, collate_fn=custom_collate
        )
    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            pin_memory=args.pin_memory
        )

    # --- Model Path Logic ---
    model_filename = f"best_model_risk_prediction_id-{args.id_training}.pth"



    path_model_risk = os.path.join(args.path_out_dir, model_filename)
    logg_filename = f"test_risk_prediction_training_id_{args.id_training}.log"
    path_logger = os.path.join(args.path_test_folder, logg_filename)

    if accelerator.is_main_process:
        os.makedirs(args.path_test_folder, exist_ok=True)
        print("Model path:", path_model_risk)
        print("Logger path:", path_logger)

    # --- Run Evaluation ---
    test_risk(
        args=args,
        test_loader=test_loader,
        path_model=path_model_risk,
        out_dir= args.path_out_dir,
        path_logger=path_logger,
        accelerator=accelerator,
    )


if __name__ == "__main__":
    main()
