import os
import torch, torchvision
import torch.optim as optim
import torchvision.utils as utils
import torch.autograd as autograd
import torch.utils.data as dset
from tensorboardX import SummaryWriter
from tqdm import tqdm
import datetime

from model import Generator, Discriminator
import data_utils


class Wgan_runner():
    def __init__(self, config, args):
        self.config = config
        self.args = args
    
    def set_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            optimizer = optim.Adam(parameters, lr=self.config.optim.lr,
                betas=[self.config.optim.beta1, self.config.optim.beta2])
        elif self.config.optim.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(parameters, lr=self.config.optim.lr)
        elif self.config.optim.optimizer == 'SGD':
            optimizer = optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotimplementedError('Optimizer {} is not understood.'.format(
                self.config.optim.optimizer
            ))
        return optimizer
    
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        eta = torch.FloatTensor(self.config.training.batch_size,
                                1, 1, 1).uniform_(0,1)
        eta = eta.expand(self.config.training.batch_size,
                         real_data.size(1),
                         real_data.size(2),
                         real_data.size(3)).to(self.device)
        interpolates = eta * real_data + ((1 - eta) * fake_data)
        interpolates = interpolates.to(self.device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                              create_graph=True, retain_graph=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()*self.config.model.Lambda
        return gradient_penalty
        
    
    def train(self, check_point=None):
        self.device = 'cuda:0' if self.config.training.use_gpu else 'cpu'

        train_data, _ = data_utils.load(self.config.data.dataset)
        image_loader = data_utils.InfiniteLoader(dset.DataLoader(
            dataset=train_data,
            batch_size=self.config.training.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=2
        ))

        if check_point is not None:
            try:
                states = torch.load(check_point)
            except FileNotFoundError:
                print("Check point is not Found...\n")
            netD = Discriminator(self.config.data.channels).to(self.device)
            netG = Generator(self.config.data.channels).to(self.device)
            netD.load_state_dict(states[0])
            netG.load_state_dict(states[1])
            optD = self.set_optimizer(netD.parameters())
            optG = self.set_optimizer(netG.parameters())
            optD.load_state_dict(states[2])
            optG.load_state_dict(states[3])
        else:
            netD = Discriminator(self.config.data.channels).to(self.device)
            netG = Generator(self.config.data.channels).to(self.device)
            optD = self.set_optimizer(netD.parameters())
            optG = self.set_optimizer(netG.parameters())

        writer = SummaryWriter(log_dir=self.config.training.log_dir)
        one = torch.tensor(1., dtype=torch.float).to(self.device)
        mone = (one * -1).to(self.device)
        tbar = tqdm(range(self.config.training.max_iter))

        for g_iter in tbar:
        
            for p in netD.parameters():
                p.requires_grad = True
            
            d_loss_real = 0.
            d_loss_fake = 0.
            Wasserstein_D = 0.
            # netD optimization
            for d_iter in range(self.config.training.critic_iter):
                netD.zero_grad()

                real_images, _ = next(image_loader)
                real_images = real_images.to(self.device)
                z = torch.randn([self.config.training.batch_size,
                                 self.config.model.hidden_dim,
                                 1,
                                 1]).to(self.device)

                d_loss_real = netD(real_images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                fake_images = netG(z)
                d_loss_fake = netD(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)
                
                gradient_penalty = self.calc_gradient_penalty(netD, real_images, fake_images)
                gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake

                optD.step()
            
            # netG optimization
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            
            z = torch.randn([self.config.training.batch_size,
                                 self.config.model.hidden_dim,
                                 1,
                                 1]).to(self.device)
            fake_images = netG(z)

            g_loss = netD(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)   
            
            optG.step()

            if (g_iter+1) % self.config.training.minitor_iter == 0:
                # Using Tensorboard to log training
                log = {
                    'wasserstein_distance': Wasserstein_D.item(),
                    'loss_D': d_loss.item(),
                    'loss_G': g_loss.item(),
                    'loss_D_real': d_loss_real.item(),
                    'loss_G_fake': d_loss_fake.item()
                }
                info = "[{}/{}], loss_D: {:.3f}, loss_G: {:.3f}, W_distance: {:.3f}".format(
                    g_iter+1, self.config.training.max_iter, d_loss.item(), -g_loss.item(), Wasserstein_D.item()
                )
                tbar.set_description(info)

                for key, value in log.items():
                    writer.add_scalar(key, value, g_iter + 1)
            
            if (g_iter+1) % self.config.training.save_iter == 0:

                real_images = real_images.mul(0.5).add(0.5).cpu()
                z = torch.randn([self.config.training.batch_size,
                                 self.config.model.hidden_dim,
                                 1,
                                 1]).to(self.device)
                samples = netG(z)
                samples = samples.mul(0.5).add(0.5).cpu()
                grid = utils.make_grid(samples)
                utils.save_image(grid, os.path.join(self.args.image_path, self.config.data.dataset, "iters_%d.png" % (g_iter+1)))

                image_log = {
                    'real_image': real_images,
                    'generated_image': samples
                }
                for key, value in image_log.items():
                    writer.add_images(key, value, g_iter + 1)

                states = [
                    netD.state_dict(),
                    netG.state_dict(),
                    optD.state_dict(),
                    optG.state_dict()
                ]
                torch.save(states, os.path.join(self.config.training.check_point, 'checkpoint_{}.pth'.format(g_iter+1)))
                torch.save(states, os.path.join(self.config.training.check_point, 'checkpoint.pth'))


    def sample(self, sample_size = 64):
        states = torch.load(os.path.join(self.config.training.check_point, 'checkpoint.pth'), map_location='cpu')
        netG = Generator(self.config.data.channels)

        netG.load_state_dict(states[1])
        netG.eval()

        if not os.path.exists(self.args.image_path):
            os.makedirs(self.args.image_path)

        z = torch.randn([sample_size,
                        self.config.model.hidden_dim,
                        1,
                        1])
        samples = netG(z)
        samples = samples.mul(0.5).add(0.5)
        grid = utils.make_grid(samples)
        time_stamp = datetime.datetime.now().timestamp()
        utils.save_image(grid, os.path.join(self.args.image_path, 
                                            self.config.data.dataset,
                                            "generated_image_{}.png".format(
                                                time_stamp
                                            )))