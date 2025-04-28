import numpy as np
import torch
import time
import scattering as st
from cross_scattering.cross_scattering2d import CrossScattering2d
from scattering.Scattering2d import get_scattering_index

class PPDVSyn:
    def __init__(self, M, N, J, L=4, estimator='s_cov', device='gpu', seed=None,
                 wavelets='morlet', C11_criteria=None, normalization='P00', weight=None):
        if not torch.cuda.is_available(): device='cpu'
        np.random.seed(seed)
        if C11_criteria is None:
            self.C11_criteria = 'j2>=j1'
        
        # define calculator
        self.J = J; self.L = L
        self.device = device    
        self.weight = weight
        self.st_calc = CrossScattering2d(M, N, J, L, device, wavelets, weight=weight)


        # define estimator function
        select_and_index = get_scattering_index(self.J, self.L, num_field=2)
        if estimator == 's_cov':
            self.estimator_func = lambda x: self.st_calc.scattering_cov(
                x, use_ref=True, C11_criteria=C11_criteria, 
                normalization=normalization)['for_synthesis']
            def func_s_2field(target, image):
                result = self.st_calc.scattering_cross_cov(
                    target, image, normalization='P00', use_ref=True
                )
                N_image = target.shape[0]
                C00 = result['Corr00_iso']
                C01 = result['C01_iso'][:,select_and_index['select_2_iso']]
                C11 = result['Corr11_iso'][:,select_and_index['select_3_iso']]
                for_synthesis_iso = torch.cat((
                    C00.real, 
                    C01.reshape((N_image, -1)),
                    C11.reshape((N_image, -1)),
                ), dim=-1)
                return for_synthesis_iso
            self.estimator_func_cov = func_s_2field
        if estimator=='s_cov_iso':
            self.estimator_func = lambda x: self.st_calc.scattering_cov(
                x, use_ref=True, C11_criteria=C11_criteria, 
                normalization=normalization)['for_synthesis_iso']
            def func_s_2field(target, image):
                result = self.st_calc.scattering_cross_cov(
                    target, image, normalization='P00', use_ref=True
                )
                N_image = target.shape[0]
                C00 = result['Corr00_iso']
                C01 = result['C01_iso'][:,select_and_index['select_2_iso']]
                C11 = result['Corr11_iso'][:,select_and_index['select_3_iso']]
                for_synthesis_iso = torch.cat((
                    C00.real, 
                    C01.reshape((N_image, -1)),
                    C11.reshape((N_image, -1)),
                ), dim=-1)
                return for_synthesis_iso
            self.estimator_func_cov = func_s_2field
        if estimator=='st_corr':
            def func_s(image, image_b):
                result = self.st_calc.scattering_cross_cov(
                    image, image_b, C11_criteria=C11_criteria,
                    normalization=normalization
                )
                return result['for_synthesis']
            self.estimator_func = func_s


    # define synthesis function
    def synthesis(self, target_channel, ref_channel, image_init, optim_algorithm='LBFGS', 
                  steps=300, learning_rate=0.2, print_each_step=False):
        
        # formating input images
        if type(target_channel)==np.ndarray:
            target_channel = torch.from_numpy(target_channel)
        if type(ref_channel)==np.ndarray:
            ref_channel = torch.from_numpy(ref_channel)
        if type(image_init)==np.ndarray:
            image_init = torch.from_numpy(image_init)
        if type(self.weight)==np.ndarray:
            self.weight = torch.from_numpy(self.weight)
        if self.device=='gpu':
            image_init = image_init.cuda()
            target_channel = target_channel.cuda()
            ref_channel = ref_channel.cuda()
            self.weight = self.weight.cuda()


        # define optimizable image model
        class OptimizableImage(torch.nn.Module):
            def __init__(self, input_init, Fourier=False):
                # super(OptimizableImage, self).__init__()
                super().__init__()
                self.param = torch.nn.Parameter( input_init )
                
                if Fourier: 
                    self.image = torch.fft.ifftn(
                        self.param[0] + 1j*self.param[1],
                        dim=(-2,-1)).real
                else: self.image = self.param
        
        image_model = OptimizableImage(self.image_transform_inverse(image_init, target_channel))


        # define optimizer
        if optim_algorithm   =='Adam':
            optimizer = torch.optim.Adam(image_model.parameters(), lr=learning_rate)
        elif optim_algorithm =='NAdam':
            optimizer = torch.optim.NAdam(image_model.parameters(), lr=learning_rate)
        elif optim_algorithm =='SGD':
            optimizer = torch.optim.SGD(image_model.parameters(), lr=learning_rate)
        elif optim_algorithm =='Adamax':
            optimizer = torch.optim.Adamax(image_model.parameters(), lr=learning_rate)
        elif optim_algorithm =='LBFGS':
            optimizer = torch.optim.LBFGS(image_model.parameters(), lr=learning_rate, 
                max_iter=1, max_eval=None, 
                tolerance_grad=1e-8, tolerance_change=1e-10, 
                history_size=min(steps//2, 100), line_search_fn=None
            )
        
        # optimize
        def closure():
            optimizer.zero_grad()
            loss, prior_loss, cross_loss = 0, 0, 0
            # compute loss terms for the physical components
            image_residual = target_channel-self.image_transform(image_model.image, target_channel)
            image_syn = torch.stack([self.image_transform(image_model.image, target_channel), image_residual], dim=0)


            # compute loss terms
            self.st_calc.add_ref(ref_channel)
            self.st_calc.add_ref_ab(target_channel, ref_channel)
            estimator_model = self.estimator_func(image_syn[0])
            estimator_target = self.estimator_func(ref_channel)
            prior_loss += self.prior_loss(estimator_model, estimator_target)
            cross_loss += self.cross_loss(self.estimator_func_cov(image_syn[1], ref_channel))
            loss = prior_loss + cross_loss


            if print_each_step:
                if (i%10==0 or i%10==-1):
                    print(f"Step {i}: Total Loss: {loss:.4f}, Prior Loss: {prior_loss:.4f}, Cross Loss: {cross_loss:.4f}")
                    # if loss_mode == 'full':
                    #     print("Cross Loss per component: ", [np.round(loss.item(), decimals=4) for loss in cross_loss_per_component])
            
            loss.backward()
            
            return loss


         # optimize
        t_start = time.time()
        cmin, cmax = target_channel-target_channel, target_channel
        if optim_algorithm =='LBFGS':
            for i in range(steps):
                optimizer.step(closure)
                # for p in image_model.parameters():
                #     p.data.clamp_(min=cmin, max=cmax)
        else:
            for i in range(steps):
                # print('step: ', i)
                optimizer.step(closure)
                for p in image_model.parameters():
                    p.data.clamp_(min=cmin, max=cmax)
        t_end = time.time()
        print('time used: ', t_end - t_start, 's')


        image_residual = target_channel-self.image_transform(image_model.image, target_channel)
        image_syn = torch.stack([self.image_transform(image_model.image, target_channel), image_residual], dim=0)
        
        return image_syn.cpu().detach().numpy()


    # define prior loss function
    def prior_loss(self, model, target):
        return ((model - target)**2).mean()*1e3
    
    # define cross loss function
    def cross_loss(self, metric):
        return ((metric)**2).mean()*1e3
    
    def image_transform(self, image_model, image_original):
        res = image_original * torch.sigmoid(image_model)
        return res
    
    def image_transform_inverse(self, image_model, image_original):
        # Invert the sigmoid transformation to recover the model parameters
        eps = 1e-7  # Small constant to prevent log(0)
        transformed = torch.clamp(image_model / (image_original + eps), eps, 1-eps)
        res = torch.log(transformed / (1 - transformed))
        return res
