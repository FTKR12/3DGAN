import sys
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import interpolate

from models.build_model import build_netG, build_netD
from data.customdataset import CustomDataset
from models.losses import gdloss
from utils.util import new_state_dict
from utils.logger import setup_logger
from utils.seed import set_seed
from options import Options
from tqdm import tqdm

def main(opt, logger):

    # dataloader
    opt.phase = 'train'
    train_dataset = CustomDataset(opt)
    logger.info(f'Training Image Numbers: {train_dataset.img_size}')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers))
    opt.phase = 'valid'
    valid_dataset = CustomDataset(opt)
    logger.info(f'Validation Image Numbers: {valid_dataset.img_size}')
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=int(opt.workers))
    opt.phase = 'train'

    # model
    generator = build_netG(opt)
    discriminator, target_real, target_fake = build_netD(opt)

    # use gpu or cpu
    if opt.gpu_ids != '-1':
        num_gpus = len(opt.gpu_ids.split(','))
    else:
        num_gpus = 0
    logger.info(f'Number of GPU: {num_gpus}')

    # loss
    adversarial_criterion = nn.MSELoss()  # nn.BCELoss()

    # set * on cuda, and dataparallel
    if (opt.gpu_ids != -1) & torch.cuda.is_available():
        use_gpu = True
        generator.cuda()
        discriminator.cuda()
        adversarial_criterion.cuda()
        target_real = target_real.cuda()
        target_fake = target_fake.cuda()
        if num_gpus > 1:
            generator = nn.DataParallel(generator)
            discriminator = nn.DataParallel(discriminator)

    # optimizer
    optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR, weight_decay=1e-4)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR, weight_decay=1e-4)
    StepLR_G = torch.optim.lr_scheduler.StepLR(optim_generator, step_size=10, gamma=0.85)
    StepLR_D = torch.optim.lr_scheduler.StepLR(optim_discriminator, step_size=10, gamma=0.85)

    # training
    logger.info('start training')
    min_loss = 1000.0
    for epoch in tqdm(range(opt.nEpochs)):
        logger.info(f'EPOCH: {epoch}')
        # init loss
        train_mean_generator_adversarial_loss = 0.0
        train_mean_generator_l2_loss = 0.0
        train_mean_generator_gdl_loss = 0.0
        train_mean_generator_total_loss = 0.0
        train_mean_discriminator_loss = 0.0

        for i, data in enumerate(train_dataloader):
            # get input data
            high_real_patches = data['high_img_patches']  # [batch_size,num_patches,C,D,H,W]
            for k in range(0, opt.num_patches):
                high_real_patch = high_real_patches[:, k]  # [BCDHW]
                low_patch = interpolate(high_real_patch, scale_factor=0.5)
                if use_gpu:
                    high_real_patch = high_real_patch.cuda()
                    # generate fake data
                    high_gen = generator(low_patch.cuda())
                else:
                    high_gen = generator(low_patch)

                ######### Train discriminator #########
                discriminator.zero_grad()

                discriminator_loss = 0.5 * adversarial_criterion(discriminator(high_real_patch), target_real) + \
                                    0.5 * adversarial_criterion(discriminator(high_gen.detach()), target_fake)

                train_mean_discriminator_loss += discriminator_loss
                discriminator_loss.backward()
                optim_discriminator.step()

                ######### Train generator #########
                generator.zero_grad()

                generator_gdl_loss = opt.gdl * gdloss(high_real_patch, high_gen)
                train_mean_generator_gdl_loss += generator_gdl_loss

                generator_l2_loss = nn.MSELoss()(high_real_patch, high_gen)
                train_mean_generator_l2_loss += generator_l2_loss

                generator_adversarial_loss = adversarial_criterion(discriminator(high_gen), target_real)
                train_mean_generator_adversarial_loss += generator_adversarial_loss

                generator_total_loss = generator_gdl_loss + generator_l2_loss + opt.advW * generator_adversarial_loss
                train_mean_generator_total_loss += generator_total_loss

                generator_total_loss.backward()
                optim_generator.step()

        StepLR_G.step()
        StepLR_D.step()

        train_mean_generator_adversarial_loss /= len(train_dataloader)
        train_mean_discriminator_loss /= len(train_dataloader)
        train_mean_generator_l2_loss /= len(train_dataloader)
        train_mean_generator_gdl_loss /= len(train_dataloader)
        train_mean_generator_total_loss /= len(train_dataloader)

        logger.info('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (GDL/L2/Adv/Total): %.4f/%.4f/%.4f/%.4f\n' % (epoch, opt.nEpochs, i, len(train_dataloader), train_mean_discriminator_loss, train_mean_generator_gdl_loss, train_mean_generator_l2_loss, train_mean_generator_adversarial_loss, train_mean_generator_total_loss))

        # validating
        # init loss
        valid_mean_generator_l2_loss = 1000.0

        with torch.no_grad():
            for i, data in enumerate(valid_dataloader):
                # get input data
                high_real_patches = data['high_img_patches']  # [batch_size,num_patches,C,D,H,W]
                for k in range(0, opt.num_patches):
                    high_real_patch = high_real_patches[:, k]  # [BCDHW]
                    low_patch = interpolate(high_real_patch, scale_factor=0.5)
                    if use_gpu:
                        high_real_patch = high_real_patch.cuda()
                        # generate fake data
                        high_gen = generator(low_patch.cuda())
                    else:
                        high_gen = generator(low_patch)

                    generator_l2_loss = nn.MSELoss()(high_real_patch, high_gen)
                    valid_mean_generator_l2_loss += generator_l2_loss

            valid_mean_generator_l2_loss /= len(train_dataloader)

        
        if min_loss > valid_mean_generator_l2_loss:
            # Do checkpointing
            torch.save(generator.state_dict(), '%s/g.pth' % opt.checkpoints_dir)
            torch.save(discriminator.state_dict(), '%s/d.pth' % opt.checkpoints_dir)
            logger.info('Model Saved.')
            min_loss = valid_mean_generator_l2_loss

        logger.info('\r[%d/%d][%d/%d] Generator_Loss (L2): %.4f\n' % (epoch, opt.nEpochs, i, len(train_dataloader), valid_mean_generator_l2_loss))


if __name__ == '__main__':
    # options
    opt = Options().parse()
    opt.phase = 'train'

    # make directories
    print(opt)
    os.makedirs(opt.result_dir, exist_ok=True)
    os.makedirs(opt.checkpoints_dir, exist_ok=True)

    # seed
    set_seed(opt.seed)
    
    # logger
    logger = setup_logger('Lung Generation', save_dir=opt.result_dir)
    logger.info(str(opt).replace(',', '\n'))

    # train
    main(opt, logger)