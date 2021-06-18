import argparse
import math
import os

import torch
import torchvision
from torch import optim
from tqdm import tqdm

from model import Generator
from utils import ensure_checkpoint_exists

import torch
import clip


class CLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def main(args):
    ensure_checkpoint_exists(args.ckpt)
    text_inputs = torch.cat([clip.tokenize(args.description)]).cuda()
    os.makedirs(args.results_dir, exist_ok=True)

    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)

    if args.ckpt2 is not None:
        g_ema2 = Generator(args.stylegan_size, 512, 8)
        g_ema2.load_state_dict(torch.load(args.ckpt2)["g_ema"], strict=False)
        g_ema2.eval()
        g_ema2 = g_ema2.cuda()
        mean_latent2 = g_ema2.mean_latent(4096)
        

    if args.latent_path:
        latent_code_init = torch.load(args.latent_path).cuda()
    else:
        if args.seed is not None:
            torch.manual_seed(args.seed)
        latent_ran = torch.randn(1, args.latent_dim, 512, device=args.device)
        latent_code_init = g_ema.get_latent(latent_ran).detach().clone()

    latent = latent_code_init.detach().clone()
    latent.requires_grad = True

    clip_loss = CLIPLoss(args)

    optimizer = optim.Adam([latent], lr=args.lr)

    pbar = tqdm(range(args.step))

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
  
        img_gen, _ = g_ema([latent], input_is_latent=True,
                           truncation = args.truncation, truncation_latent = mean_latent,
                           randomize_noise=False)
        if args.ckpt2 is not None:
            img_gen2, _ = g_ema2([latent], input_is_latent=True,
                           truncation = args.truncation, truncation_latent = mean_latent2,
                           randomize_noise=False)

        c_loss = clip_loss(img_gen, text_inputs)
        l2_loss = ((latent_code_init - latent) ** 2).sum()
        loss = c_loss + args.l2_lambda * l2_loss

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        pbar.set_description(
            (
                f"loss: {loss.item():.4f};"
            )
        )
        if args.save_intermediate_image_every > 0 and i % args.save_intermediate_image_every == 0:
            with torch.no_grad():
                img_gen, _ = g_ema([latent], input_is_latent=True,
                                    truncation = args.truncation, truncation_latent = mean_latent,
                                    randomize_noise=False)
                if args.ckpt2 is not None:
                    img_gen2, _ = g_ema2([latent], input_is_latent=True,
                                          truncation = args.truncation, truncation_latent = mean_latent2,
                                          randomize_noise=False)
                            
            # Save Image
            
            if args.ckpt2 is not None:
                
                torchvision.utils.save_image(
                    torch.cat([img_gen, img_gen2], 0),
                    f"{args.results_dir}/{str(i).zfill(5)}.png",
                    normalize=True,
                    range=(-1, 1),
                    nrow=1,
                )
            else :
                torchvision.utils.save_image(img_gen, f"{args.results_dir}/{str(i).zfill(5)}.png", normalize=True, range=(-1, 1))


    with torch.no_grad():
        img_orig, _ = g_ema([latent_code_init], input_is_latent=True,
                           truncation = args.truncation, truncation_latent = mean_latent,
                           randomize_noise=False)

    final_result = torch.cat([img_orig, img_gen], 0)
    if args.ckpt2 is not None:
        final_result = torch.cat([img_orig, img_gen, img_gen2], 0)


    return final_result, latent_code_init, latent



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, default="a person with purple hair", help="the text that guides the editing/generation")
    parser.add_argument("--ckpt", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt", help="pretrained StyleGAN2 weights")
    parser.add_argument("--ckpt2", type=str, default=None, help="pretrained StyleGAN2 weights")
    
    parser.add_argument("--stylegan_size", type=int, default=256, help="StyleGAN resolution")
    parser.add_argument("--latent_dim", type=int, default=14, help="StyleGAN latent dimension")
    parser.add_argument("--seed", type=int, default=14, help="StyleGAN latent seed")

    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--step", type=int, default=300, help="number of optimization steps")
    parser.add_argument("--l2_lambda", type=float, default=0.008, help="weight of the latent distance (used for editing only)")
    parser.add_argument("--latent_path", type=str, default=None, help="starts the optimization from the given latent code if provided. Otherwose, starts from"
                                                                      "the mean latent in a free generation, and from a random one in editing. "
                                                                      "Expects a .pt format")
    parser.add_argument("--truncation", type=float, default=0.7, help="used only for the initial latent vector, and only when a latent code path is"
                                                                      "not provided")
    parser.add_argument("--save_intermediate_image_every", type=int, default=20, help="if > 0 then saves intermidate results during the optimization")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")


    args = parser.parse_args()

    result_image = main(args)

    torchvision.utils.save_image(result_image.detach().cpu(), 
                                os.path.join(args.results_dir, "final_result.jpg"), 
                                normalize=True, scale_each=True, range=(-1, 1))



