when i run https://github.com/hkzhang-git/ParC-Net/tree/main code to train my custom data to do some classification,
i can't convert pth to onnx.
and i found the code and do some modifies to solve the problem.

step:
1. copy the class "gcc_ca_mf_block_onnx" from     EdgeFormer/EdgeFormer/cvnets/modules/edgeformer_block.py  to your code ParC-Net/cvnets/modules/edgeformer_block.py
2. add "gcc_ca": gcc_ca_mf_block_onnx in your code(ParC-Net/cvnets/modules/edgeformer_block.py), likes:   
block_dict={
    'bkc': bkc_mf_block,
    'bkc_ca': bkc_ca_mf_block,
    'gcc': gcc_mf_block,
    'gcc_ca': gcc_ca_mf_block,
    'gcc_dk': gcc_dk_mf_block,
    'gcc_dk_ca': gcc_dk_ca_mf_block,
    'gcc_onnx':gcc_ca_mf_block_onnx
}
3. copy model_convert2onnx.py to your ParC-Net path, run:
python model_conver2onnx.py
4. to check your onnx is working
   python -c "import onnx; m=onnx.load('edgeformer.onnx'); onnx.checker.check_model(m)"
