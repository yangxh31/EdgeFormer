import torch
from torch import nn
from EdgeFormer.cvnets.layers import GlobalPool

from EdgeFormer.cvnets.models.classification.edgeformer import edgeformer
import torch
import torch.nn.functional as F
import multiprocessing

from EdgeFormer.cvnets import get_model
from EdgeFormer.utils import logger
from EdgeFormer.utils.common_utils import device_setup
from EdgeFormer.options.opts import get_eval_arguments, get_training_arguments
from EdgeFormer.options.utils import load_config_file

file_name='path'

opts = get_training_arguments(False)


node_rank = getattr(opts, "ddp.rank", 0)
if node_rank < 0:
    logger.error('--rank should be >=0. Got {}'.format(node_rank))



# No of data workers = no of CPUs (if not specified or -1)
n_cpus = multiprocessing.cpu_count()

# adjust the batch size
train_bsize = 32
val_bsize = 32
setattr(opts, "dataset.val_batch_size0", val_bsize)
setattr(opts, "dev.device_id", None)
setattr(opts,"common.config_file","./EdgeFormer/config/classification/edgeformer/edgeformer_s.yaml")
setattr(opts,"dataset.category",'classification')
# print(vars(opts))
opts = load_config_file(opts)
# print(vars(opts))
setattr(opts,"ddp.rank",1)
setattr(opts,"model.classification.name",'edgeformer')


num_gpus = getattr(opts, "dev.num_gpus", 0)  # defaults are for CPU
dev_id = getattr(opts, "dev.device_id", torch.device('cpu'))
device = getattr(opts, "dev.device", torch.device('cpu'))
is_distributed = getattr(opts, "ddp.use_distributed", False)



model = edgeformer(opts)



# model.load_state_dict(torch.load(f'../{file_name}.pt', map_location='cpu'))

setattr(opts,"model.classification.edge.kernel","gcc_onnx")

model_onnx = edgeformer(opts)

model.eval()
model_onnx.eval()

def move_weights_dy2fr(model_dk: nn.Module,
                       model_fr: nn.Module):
    
    """
    model_dk: dynamic kernel model with trained weights
    model_fr: frozen, exportable model with random initialized weights

    Two stage process:
        1. transfer state dict for all parameters not gcc
        2. manually move kernel tensor from dk to conv weight in fr
    """

    model_fr.load_state_dict(model_dk.state_dict(), strict=False)

    for layer in ['layer_3','layer_4','layer_5']:
        block_count = len(getattr(model_dk,layer)[1].spatial_global_rep)
        for block_iter in range(block_count-1):
            for c_name,k_name,b_name in zip(['conv_1_H','conv_1_W','conv_2_H','conv_2_W'],['meta_kernel_1_H', 'meta_kernel_1_W', 'meta_kernel_2_H', 'meta_kernel_2_W'],['meta_1_H_bias', 'meta_1_W_bias', 'meta_2_H_bias', 'meta_2_W_bias']):
                weights = getattr(getattr(model_dk,layer)[1].spatial_global_rep[block_iter],k_name)
                bias = getattr(getattr(model_dk,layer)[1].spatial_global_rep[block_iter],b_name)


                if 'H' in c_name:
                    int_shape = [weights.shape[2]//2,1]
                else:
                    int_shape = [1,weights.shape[3]//2]
                weights = F.interpolate(weights, int_shape, mode='bilinear', align_corners=True)



                getattr(getattr(model_fr,layer)[1].spatial_global_rep[block_iter],c_name).weight = nn.parameter.Parameter(weights)
                getattr(getattr(model_fr,layer)[1].spatial_global_rep[block_iter],c_name).bias = bias
    return model_fr
x = torch.rand(1,3,128,128)
old_out = model_onnx(x)

new_model = move_weights_dy2fr(model,model_onnx)


out = model(x)
new_out = new_model(x)

print(f"old versus new {all(old_out[0]==new_out[0])}")
print(f"dy versus fr {all(out[0] == new_out[0])}")

x = torch.randn(1, 3, 128, 128,requires_grad=True)
model_onnx.eval()


torch_out = new_model(x)

# Export the model
torch.onnx.export(new_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  f"{file_name}.onnx",   # where to save the model (can be a file or file-like object)
                  opset_version=12,          # the ONNX version to export the model to
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'sequence'}, 'output': {0: 'sequence'}}
 )
