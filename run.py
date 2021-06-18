import os
import torch
import argparse
from argparse import Namespace

def prepare_data(dataset_folder, zip_file=None, target_size=256):

    # Unzip
    if zip_file is not None:
        os.system(f'unzip {zip_file} -d "/{zip_file}"')
        os.system(f'rm {zip_file}')
    
    # prepare data
    os.system(f'python prepare_data.py --out {dataset_folder}/LMDB --size {target_size} {dataset_folder}')

def download_pretrained_model(DownLoad_All=True, file=''):
    from utils import download_pretrained_model 
    
    if DownLoad_All:
        download_pretrained_model()
    else:
        download_pretrained_model(False, "ffhq256.pt")

def project(encoder = True, img='00006.jpg'):
    if encoder:
        os.system(f'python projector_factor.py --ckpt=/networks/ffhq256.pt --e_ckpt=/networks/encoder_ffhq.pt \
                            --files=/asset/ffhq-sample/{img}')
    else:
        os.system(f'python projector_factor.py --ckpt=/networks/ffhq256.pt \
                            --files=/asset/ffhq-sample/{img}')   

def generate_using_styleclip(description, seed=100, 
                            network1="/networks/ffhq256.pt",
                            network2="networks/ffhq256.pt",
                            latent_path=None, optimization_steps=300, truncation = 0.7,
                            l2_lambda = 0.004, result_dir = "asset/results_styleclip",
                            device = 'cuda',
                            number_of_step = 5 ,
                            strength = 1.5,
                            swap = True, 
                            swap_layer_num = 1):

    # -----------------------------
    args = {
        "seed" : seed,
        "description": description,
        "ckpt": network1,
        "ckpt2": network2,
        "stylegan_size": 256,
        "latent_dim" : 14,
        "lr_rampup": 0.05,
        "lr": 0.1,
        "step": optimization_steps,
        "l2_lambda": l2_lambda,
        "latent_path": latent_path,
        "truncation": truncation,
        "device" : device,
        "results_dir": result_dir,
    }
    
    from run_optimization import main
    final_result, latent_init1, latent_fin1 = main(Namespace(**args))

    # ---------------
    # Generator
    # ---------------
    from model import Generator

    # Generator1
    network1 = torch.load(network1)

    generator1 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    generator1.load_state_dict(network1["g_ema"], strict=False)

    trunc1 = generator1.mean_latent(4096)

    # Generator2
    network2 = torch.load(network2)

    generator2 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    generator2.load_state_dict(network2["g_ema"], strict=False)

    trunc2 = generator2.mean_latent(4096)

    # ---------------
    # Interpolation
    # ---------------

    latent_interp = torch.zeros(number_of_step, latent_init1.shape[1], latent_init1.shape[2]).to(device)

    with torch.no_grad():
        for j in range(number_of_step):

            latent_interp[j] = latent_init1 + strength * (latent_fin1-latent_init1) * float(j/(number_of_step-1))

            imgs_gen1, save_swap_layer = generator1([latent_interp],
                                    input_is_latent=True,                                     
                                    truncation=0.7,
                                    truncation_latent=trunc1,
                                    swap=swap, swap_layer_num=swap_layer_num,
                                    randomize_noise=False)
            imgs_gen2, _ = generator2([latent_interp],
                                    input_is_latent=True,                                     
                                    truncation=0.7,
                                    swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
                                    truncation_latent=trunc2)

    im1 = torch.cat([img_gen for img_gen in imgs_gen1], dim=2)
    im2 = torch.cat([img_gen for img_gen in imgs_gen2], dim=2)
    result = torch.cat([im1, im2], dim=1)
    return result # if you want to show image :: `imshow(tensor2image(result))`

def generate_using_latent_mixing(seed1=100, seed2=200,
                            network1="/networks/ffhq256.pt",
                            network2="networks/ffhq256.pt",
                            latent_mixing1=10, latent_mixing2=10,
                            latent_path=None, optimization_steps=300, truncation = 0.7,
                            l2_lambda = 0.004, result_dir = "asset/results_styleclip",
                            device = 'cuda',
                            number_of_step = 5 ,
                            strength = 1.5,
                            swap = True, 
                            swap_layer_num = 1):


    # ----------------------
    # Source Images (FFHQ)
    # ----------------------

    from model import Generator

    # Genearator1
    network1 = torch.load(network1)

    generator1 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    generator1.load_state_dict(network1["g"], strict=False)
    trunc1 = generator1.mean_latent(4096)

    # latent1
    torch.manual_seed(seed1)
    r_latent1 = torch.randn(1, 14, 512, device=device)
    latent1 = generator1.get_latent(r_latent1)

    # latent2
    torch.manual_seed(seed2)
    r_latent2 = torch.randn(1, 14, 512, device=device)
    latent2 = generator1.get_latent(r_latent2)

    # latent mixing
    latent3 = torch.cat([latent1[:,:latent_mixing1,:], latent2[:,latent_mixing1:,:]], dim = 1)
    latent4 = torch.cat([latent1[:,:,:latent_mixing2], latent2[:,:,latent_mixing2:]], dim = 2)

    # Latent !
    latent = torch.cat([latent1, latent2, latent3, latent4], dim = 0)

    # generate image
    img1, save_swap_layer = generator1(
        [latent],
        input_is_latent=True,
        truncation=truncation,
        truncation_latent=trunc1,
        swap=swap, swap_layer_num=swap_layer_num,
    )

    # =================================================

    # ----------------------
    # Target Images (Cartoon)
    # ----------------------

    # Genearator2
    network2 = torch.load(network2)

    generator2 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    generator2.load_state_dict(network2["g"], strict=False)
    trunc2 = generator2.mean_latent(4096)

    # generate image
    img2, _ = generator2(
        [latent],
        input_is_latent=True,
        truncation=truncation,
        truncation_latent=trunc1,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
    )

    # return

    ffhq = torch.cat([img1[0], img1[1], img1[2], img1[3]], dim=2)
    cartoon = torch.cat([img2[0], img2[1], img2[2], img2[3]], dim=2)

    return torch.cat([ffhq, cartoon], dim = 1)

def generate_using_sefa(network1="/networks/ffhq256.pt",
                        network2="/networks/ffhq256.pt",
                        factor='factor.pt',
                        index=7, degree=14, seed=116177, n_sample=5,
                        result_dir='asset/results-sefa'):

    os.system(f'python apply_factor.py --index={index} --degree={degree} --seed={seed} --n_sample={n_sample} \
                        --ckpt={network1} --ckpt2={network2} \
                        --factor={factor} --outdir={result_dir} --video')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--prepare_data", type=str, default=None)
    parser.add_argument("--zip", type=str, default=None)
    parser.add_argument("--size", type=int, default=256)


    parser.add_argument("--gif", action="store_true", help="path to the lmdb dataset")

    args = parser.parse_args()


    if args.prepare_data is not None:
        prepare_data(dataset_folder = args.prepare_data, zip_file = args.zip, target_size = args.size)
