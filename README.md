## Code optimization target
models/network_onlyattnnoir_gumbel_compensate2_simple2.py

## Measure test result / time of ICASR and SwinIR

### Install Pytorch and other stuffs (please refer to requirements.txt).

### Get DIV2K validation dataset for measuring the runtime
- HR: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
- LR: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip

### Download Pretrained models to report PSNR and SSIM
- SwinIR: https://drive.google.com/drive/folders/12UrZAg06T7l_ElqLYAtzKGc7b81Ldh-_?usp=sharing
- ICASR(Ours): https://drive.google.com/drive/folders/1Cf3wExjvVtA_bIktI8RU3C8eGfEYgUoy?usp=sharing


### Don't need to download pretrained models if not measuring PSNR, SSIM

### Change dataset path in 'main_test_clustertransformer.py' (Line 35-49)

### Move to file 'clustertest.sh'
- Use Line 2 to test ICASR (Simplified version)
- Erase '--model_path ~~~' if only need to check runtime.
- add '--bs ' for increasing batch size, '--cpu' for cpu testing


### Run code by 'sh clustertest.sh'