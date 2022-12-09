# CMGN
A Conditional Molecular Generation Net to Design Target-specific Molecules with Desired Properties

## environment
#### rdkit
#### torchtext
#### torch
#### torchvision
#### transformers
#### datasets
#### tqdm
#### wandb
#### pytorch_lightning

## download
Model weights and datasets:

https://pan.baidu.com/s/19I4_gQyWd0x1CBkNh55YiA 
nfcp


## test of molecular generation for specific-target
```bash
python test_infer_molecular_formula_mul_pr_rerank.py \
--val_folder = "exp/test"
--weights_path = "weight_smiles_decoder/PKI_100epoch_BTK_200epoch/epoch_199_loss_0.024621.pth"
--use_dict = {"fragment": True, "molecular_weight": True, "SMILES": True,
                "logP": False, "QED": True, "SA": True}
--save_path = "exp_result/BTK_2_3_PKI100btk200_allprop.cmc"
