import numpy as np
import scattering as st
import torch
from scattering.Scattering2d import Scattering2d, cut_high_k_off, get_scattering_index

class CrossScattering2d(Scattering2d):
    def __init__(self, M, N, J, L=4, device='gpu', 
        wavelets='morlet', filters_set=None, weight=None, 
        precision='single', ref=None, ref_a=None, ref_b=None,
        l_oversampling=1, frequency_factor=1
    ):
        '''
        M: int (positive)
            the number of pixels along x direction
        N: int (positive)
            the number of pixels along y direction
        J: int (positive)
            the number of dyadic scales used for scattering analysis. 
            It is at most int(log2(min(M,N))) - 1.
        L: int (positive)
            the number of orientations used for oriented wavelets; 
            or the number of harmonics used for harmonic wavelets (L=1 means only monopole is used).
        device: str ('gpu' or 'cpu')
            the device to compute
        wavelets: str
            type of wavelets, can be one of the following:
            'morlet': morlet wavelet (basically off-center gaussians in Fourier space)
            'BS'    : bump-steerable wavelets (see https://arxiv.org/pdf/1810.12136.pdf)
            'gau'   : with the same angular dependence as the bump-steerable, but the radial
                    profile as radial = (2*k/k0)**2 * np.exp(-k**2/(2 * (k0/1.4)**2)). It is 
                    similar to morlet wavelet in radial profile but has a more uniform
                    orientation coverage.
            'shannon': shannon wavelets, (top-hat profiles in Fourier space)
            'gau_harmonic': its radial profile is the same as 'gau', while the orientation 
                    profile is cyclic Fourier modes (harmonics).
        filters_set : None or dict
            if None, then it is generated automatically by the parameter provided
            otherwise, it should be a dictionary with {'psi', 'phi'}, where 'psi'
                is a torch tensor with size [J, L, M, N].
        weight: numpy array or torch tensor with size [M, N]
        precision: str ('single' or 'double')
        ref: None or numpy array or torch tensor with size [N_image, M, N] 
            the reference image used to normalize the scattering covariance. 
        ref_a, ref_b: None or numpy array or torch tensor with size [N_image, M, N]
            the reference images used to normalized the 2-field scattering covariance.
        '''
        super().__init__(M, N, J, L, device, wavelets, filters_set, 
                weight, precision, ref, ref_a, ref_b, l_oversampling, frequency_factor)
     
        # ---------------------------------------------------------------------------
    #
    # scattering cov
    #
    # ---------------------------------------------------------------------------
    def scattering_cov(
        self, data, if_large_batch=False, C11_criteria=None, 
        use_ref=False, normalization='P00', remove_edge=False,
        pseudo_coef=1, get_variance=False,
    ):
        '''
        Calculates the scattering correlations for a batch of images, including:
        orig. x orig.:     
                        P00 = <(I * psi)(I * psi)*> = L2(I * psi)^2
        orig. x modulus:   
                        C01 = <(I * psi2)(|I * psi1| * psi2)*> / factor
            when normalization == 'P00', factor = L2(I * psi2) * L2(I * psi1)
            when normalization == 'P11', factor = L2(I * psi2) * L2(|I * psi1| * psi2)
        modulus x modulus: 
                        C11_pre_norm = <(|I * psi1| * psi3)(|I * psi2| * psi3)>
                        C11 = C11_pre_norm / factor
            when normalization == 'P00', factor = L2(I * psi1) * L2(I * psi2)
            when normalization == 'P11', factor = L2(|I * psi1| * psi3) * L2(|I * psi2| * psi3)
        modulus x modulus (auto): 
                        P11 = <(|I * psi1| * psi2)(|I * psi1| * psi2)*>
        Parameters
        ----------
        data : numpy array or torch tensor
            image set, with size [N_image, x-sidelength, y-sidelength]
        if_large_batch : Bool (=False)
            It is recommended to use "False" unless one meets a memory issue
        C11_criteria : str or None (=None)
            Only C11 coefficients that satisfy this criteria will be computed.
            Any expressions of j1, j2, and j3 that can be evaluated as a Bool 
            is accepted.The default "None" corresponds to "j1 <= j2 <= j3".
        use_ref : Bool (=False)
            When normalizing, whether or not to use the normalization factor
            computed from a reference field. For just computing the statistics,
            the default is False. However, for synthesis, set it to "True" will
            stablize the optimization process.
        normalization : str 'P00' or 'P11' (='P00')
            Whether 'P00' or 'P11' is used as the normalization factor for C01
            and C11.
        remove_edge : Bool (=False)
            If true, the edge region with a width of rougly the size of the largest
            wavelet involved is excluded when taking the global average to obtain
            the scattering coefficients.
        
        Returns
        -------
        dict{'mean', 'var', 
            'P00', 'P00', 'S1', 'S1_iso', 'C01', 'C01_iso', 'C11', 'C11_iso', 
            'C11_pre_norm', 'C11_pre_norm_iso', 'P11', 'P11_iso',
            'for_synthesis', 'for_synthesis_iso', 
            'index_for_synthesis', 'index_for_synthesis_iso' 
        }:
        a dictionary containing different sets of scattering covariance coefficients.
        'P00'       : torch tensor with size [N_image, J, L] (# image, j1, l1)
            the power in each wavelet bands (the orig. x orig. term)
        'P00_iso'   : torch tensor with size [N_image, J] (# image, j1)
            'P00' averaged over the last dimension (l1)
        'S1'        : torch tensor with size [N_image, J, L] (# image, j1, l1)
            the 1st-order scattering coefficients, i.e., the mean of wavelet modulus fields
        'S1_iso'    : torch tensor with size [N_image, J] (# image, j1)
            'S1' averaged over the last dimension
        'C01'       : torch tensor with size [N_image, J, J, L, L] (# image, j1, j2, l1, l2)
            the orig. x modulus terms. Elements with j1 < j2 are all set to np.nan and not computed.
        'C01_iso'   : torch tensor with size [N_image, J, J, L] (# image, j1, j2, l2-l1)
            'C01' averaged over l1 while keeping l2-l1 constant.
        'C11'       : torch tensor with size [N_image, J, J, J, L, L, L] (# image, j1, j2, j3, l1, l2, l3)
            the modulus x modulus terms. Elements not satisfying j1 <= j2 <= j3 and the conditions
            defined in 'C11_criteria' are all set to np.nan and not computed.
        'C11_iso    : torch tensor with size [N_image, J, J, J, L, L] (# image, j1, j2, j3, l2-l1, l3-l1)
            'C11' averaged over l1 while keeping l2-l1 and l3-l1 constant.
        'C11_pre_norm' and 'C11_pre_norm_iso': pre-normalized modulus x modulus terms.
        'P11'       : torch tensor with size [N_image, J, J, L, L] (# image, j1, j2, l1, l2)
            the modulus x modulus terms with the two wavelets within modulus the same. Elements not following
            j1 <= j3 are set to np.nan and not computed.
        'P11_iso'   : torch tensor with size [N_image, J, J, L] (# image, j1, j2, l2-l1)
            'P11' averaged over l1 while keeping l2-l1 constant.
        'for_synthesis' : torch tensor with size [N_image, -1] (# image, index of coef.)
            flattened coefficients, containing mean/std, log(P00), log(S1), C01, and C11
        'for_synthesis_iso' : torch tensor with size [N_image, -1] (# image, index of coef.)
            flattened coefficients, containing mean/std, log(P00_iso), log(S1_iso), C01_iso, and C11_iso
        'index_for_synthesis' : torch tensor with size [7, -1] (index name, index of coef.)
            the index of the flattened tensor "for_synthesis", can be used to select subset of coef.
            the rows have the following meanings:
                index_type, j1, j2, j3, l1, l2, l3 = index_for_synthesis[:]
                where index_type is encoded by integers in the following way:
                    0: mean/std     1: log(P00)     2: log(S1)      
                    3: C01_real     4: C01_imag     5: C11_real     6: C011_imag
                    (7: P11)
                j range from 0 to J, l range from 0 to L.
        'index_for_synthesis_iso' : torch tensor with size [7, -1] (index name, index of coef.)
            same as 'index_for_synthesis_iso' except that it is for isotropic coefficients.
        '''
        if C11_criteria is None:
            C11_criteria = 'j2>=j1'
            
        M, N, J, L = self.M, self.N, self.J, self.L
        N_image = data.shape[0]
        filters_set = self.filters_set
        weight = self.weight
        if use_ref:
            if normalization=='P00': ref_P00 = self.ref_scattering_cov['P00']
            else: ref_P11 = self.ref_scattering_cov['P11']

        # convert numpy array input into torch tensors
        if type(data) == np.ndarray:
            data = torch.from_numpy(data)
            
        if self.device=='gpu':
            data = data.cuda()
        data_mean, data_std = (data*weight).mean((-2,-1)), (data*weight).std((-2,-1))
        data = st.whiten(data) if data.std((-2,-1)).abs().max()>1e-6 else data
        data = data * weight
        data_f = torch.fft.fftn(data, dim=(-2,-1))
        
        # initialize tensors for scattering coefficients
        P00= torch.zeros((N_image,J,L), dtype=data.dtype)
        S1 = torch.zeros((N_image,J,L), dtype=data.dtype)
        C01 = torch.zeros((N_image,J,J,L,L), dtype=data_f.dtype) + np.nan
        P11 = torch.zeros((N_image,J,J,L,L), dtype=data.dtype) + np.nan
        C11_pre_norm = torch.zeros((N_image,J,J,J,L,L,L), dtype=data_f.dtype) + np.nan
        C11 = torch.zeros((N_image,J,J,J,L,L,L), dtype=data_f.dtype) + np.nan
        
        C01_iso = torch.zeros((N_image,J,J,L), dtype=data.dtype)
        C01_reduced = torch.zeros((N_image,J,J,L), dtype=data.dtype)
        P11_iso = torch.zeros((N_image,J,J,L), dtype=data.dtype)
        C11_pre_norm_iso = torch.zeros((N_image,J,J,J,L,L), dtype=data.dtype)
        C11_iso = torch.zeros((N_image,J,J,J,L,L), dtype=data.dtype)
        C11_reduced = torch.zeros((N_image,J,J,J,L,L), dtype=data.dtype)
        
        # move torch tensors to gpu device, if required
        if self.device=='gpu':
            P00       = P00.cuda()
            S1        = S1.cuda()
            C01       = C01.cuda()
            P11       = P11.cuda()
            C11_pre_norm=C11_pre_norm.cuda()
            C11       = C11.cuda()
            C01_iso   = C01_iso.cuda()
            C01_reduced = C01_reduced.cuda()
            P11_iso   = P11_iso.cuda()
            C11_pre_norm_iso=C11_pre_norm_iso.cuda()
            C11_iso   = C11_iso.cuda()
            C11_reduced = C11_reduced.cuda()
        # calculate scattering fields
        I1 = torch.fft.ifftn(
            data_f[:,None,None,:,:] * filters_set[None,:J,:,:,:], dim=(-2,-1)
        ).abs()
        I1_f= torch.fft.fftn(I1, dim=(-2,-1))
        
        #
        if remove_edge: 
            edge_mask = self.edge_masks[:,None,:,:]
            edge_mask = edge_mask / edge_mask.mean((-2,-1))[:,:,None,None]
        else: 
            edge_mask = 1
        P00 = (I1**2 * edge_mask).mean((-2,-1))
        S1  = (I1 * edge_mask).mean((-2,-1))
#         if get_variance:
#             P00_sigma = (I1**2 * edge_mask).std((-2,-1))
#             S1_sigma  = (I1 * edge_mask).std((-2,-1))
            
        if pseudo_coef != 1:
            I1 = I1**pseudo_coef
        
        # calculate the covariance and correlations of the scattering fields
        # only use the low-k Fourier coefs when calculating large-j scattering coefs.
        for j3 in range(0,J):
            dx3, dy3 = self.get_dxdy(j3)
            I1_f_small = cut_high_k_off(I1_f[:,:j3+1], dx3, dy3) # Nimage, J, L, x, y
            data_f_small = cut_high_k_off(data_f, dx3, dy3)
            if remove_edge:
                I1_small = torch.fft.ifftn(I1_f_small, dim=(-2,-1), norm='ortho')
                data_small = torch.fft.ifftn(data_f_small, dim=(-2,-1), norm='ortho')
            wavelet_f3 = cut_high_k_off(filters_set[j3], dx3, dy3) # L,x,y
            _, M3, N3 = wavelet_f3.shape
            wavelet_f3_squared = wavelet_f3**2
            edge_dx = min(4, int(2**j3*dx3*2/M))
            edge_dy = min(4, int(2**j3*dy3*2/N))
            # a normalization change due to the cutoff of frequency space
            fft_factor = 1 /(M3*N3) * (M3*N3/M/N)**2
            for j2 in range(0,j3+1):
                I1_f2_wf3_small = I1_f_small[:,j2].view(N_image,L,1,M3,N3) * wavelet_f3.view(1,1,L,M3,N3)
                I1_f2_wf3_2_small = I1_f_small[:,j2].view(N_image,L,1,M3,N3) * wavelet_f3_squared.view(1,1,L,M3,N3)
                if remove_edge:
                    I12_w3_small = torch.fft.ifftn(I1_f2_wf3_small, dim=(-2,-1), norm='ortho')
                    I12_w3_2_small = torch.fft.ifftn(I1_f2_wf3_2_small, dim=(-2,-1), norm='ortho')
                if use_ref:
                    if normalization=='P11':
                        norm_factor_C01 = (ref_P00[:,None,j3,:] * ref_P11[:,j2,j3,:,:]**pseudo_coef)**0.5
                    if normalization=='P00':
                        norm_factor_C01 = (ref_P00[:,None,j3,:] * ref_P00[:,j2,:,None]**pseudo_coef)**0.5
                else:
                    if normalization=='P11':
                        # [N_image,l2,l3,x,y]
                        P11_temp = (I1_f2_wf3_small.abs()**2).mean((-2,-1)) * fft_factor
                        norm_factor_C01 = (P00[:,None,j3,:] * P11_temp**pseudo_coef)**0.5
                    if normalization=='P00':
                        norm_factor_C01 = (P00[:,None,j3,:] * P00[:,j2,:,None]**pseudo_coef)**0.5

                if not remove_edge:
                    C01[:,j2,j3,:,:] = (
                        data_f_small.view(N_image,1,1,M3,N3) * torch.conj(I1_f2_wf3_small)
                    ).mean((-2,-1)) * fft_factor / norm_factor_C01
                else:
                    C01[:,j2,j3,:,:] = (
                        data_small.view(N_image,1,1,M3,N3) * torch.conj(I12_w3_small)
                    )[...,edge_dx:M3-edge_dx, edge_dy:N3-edge_dy].mean((-2,-1)) * fft_factor / norm_factor_C01
                if j2 <= j3:
                    for j1 in range(0, j2+1):
                        if eval(C11_criteria):
                            if not remove_edge:
                                if not if_large_batch:
                                    # [N_image,l1,l2,l3,x,y]
                                    C11_pre_norm[:,j1,j2,j3,:,:,:] = (
                                        I1_f_small[:,j1].view(N_image,L,1,1,M3,N3) * 
                                        torch.conj(I1_f2_wf3_2_small.view(N_image,1,L,L,M3,N3))
                                    ).mean((-2,-1)) * fft_factor
                                else:
                                    for l1 in range(L):
                                        # [N_image,l2,l3,x,y]
                                        C11_pre_norm[:,j1,j2,j3,l1,:,:] = (
                                            I1_f_small[:,j1,l1].view(N_image,1,1,M3,N3) * 
                                            torch.conj(I1_f2_wf3_2_small.view(N_image,L,L,M3,N3))
                                        ).mean((-2,-1)) * fft_factor
                            else:
                                if not if_large_batch:
                                    # [N_image,l1,l2,l3,x,y]
                                    C11_pre_norm[:,j1,j2,j3,:,:,:] = (
                                        I1_small[:,j1].view(N_image,L,1,1,M3,N3) * torch.conj(
                                            I12_w3_2_small.view(N_image,1,L,L,M3,N3)
                                        )
                                    )[...,edge_dx:-edge_dx, edge_dy:-edge_dy].mean((-2,-1)) * fft_factor
                                else:
                                    for l1 in range(L):
                                    # [N_image,l2,l3,x,y]
                                        C11_pre_norm[:,j1,j2,j3,l1,:,:] = (
                                            I1_small[:,j1].view(N_image,1,1,M3,N3) * torch.conj(
                                                I12_w3_2_small.view(N_image,L,L,M3,N3)
                                            )
                                        )[...,edge_dx:-edge_dx, edge_dy:-edge_dy].mean((-2,-1)) * fft_factor
        # define P11 from diagonals of C11
        for j1 in range(J):
            for l1 in range(L):
                P11[:,j1,:,l1,:] = C11_pre_norm[:,j1,j1,:,l1,l1,:].real
        # normalizing C11
        if normalization=='P00':
            if use_ref: P = ref_P00
            else: P = P00
            #.view(N_image,J,1,1,L,1,1) * .view(N_image,1,J,1,1,L,1)
            C11 = C11_pre_norm / (
                P[:,:,None,None,:,None,None] * P[:,None,:,None,None,:,None]
            )**(0.5*pseudo_coef)
        if normalization=='P11':
            if use_ref: P = ref_P11
            else: P = P11
            #.view(N_image,J,1,J,L,1,L) * .view(N_image,1,J,J,1,L,L)
            C11 = C11_pre_norm / (
                P[:,:,None,:,:,None,:] * P[:,None,:,:,None,:,:]
            )**(0.5*pseudo_coef)
        # average over l1 to obtain simple isotropic statistics
        P00_iso = P00.mean(-1)
        S1_iso  = S1.mean(-1)
        P00_reduced = (P00*P00).mean(-1) / P00.mean(-1)
        S1_reduced = (P00*S1).mean(-1) / P00.mean(-1)
        for l1 in range(L):
            for l2 in range(L):
                C01_iso[...,(l2-l1)%L] += C01[...,l1,l2].real
                P11_iso[...,(l2-l1)%L] += P11[...,l1,l2]
                C01_reduced[...,(l2-l1)%L] += (C01[...,l1,l2].real**2)
                for l3 in range(L):
                    C11_pre_norm_iso[...,(l2-l1)%L,(l3-l1)%L]+=C11_pre_norm[...,l1,l2,l3].real
                    C11_iso[...,(l2-l1)%L,(l3-l1)%L] += C11[...,l1,l2,l3].real
                    C11_reduced[...,(l2-l1)%L,(l3-l1)%L] += (C11[...,l1,l2,l3].real**2)
        # C01_reduced /= C01_iso; C11_reduced /= C11_iso
        C01_iso /= L; P11_iso /= L; C11_pre_norm_iso /= L; C11_iso /= L
        
        # get a single, flattened data vector for_synthesis
        select_and_index        = get_scattering_index(J, L, normalization, C11_criteria)
        index_for_synthesis     = select_and_index['index_for_synthesis']
        index_for_synthesis_iso = select_and_index['index_for_synthesis_iso']
        
        for_synthesis = torch.cat((
            # (data.mean((-2,-1))/data.std((-2,-1)))[:,None],
            P00.reshape((N_image, -1)).log() if P00.sum()!=0 else P00.reshape((N_image, -1)),
            S1.reshape((N_image, -1)).log() if S1.sum()!=0 else S1.reshape((N_image, -1)),
            C01[:,select_and_index['select_2']].real, 
            C01[:,select_and_index['select_2']].imag, 
            C11[:,select_and_index['select_3']].real, 
            C11[:,select_and_index['select_3']].imag
        ), dim=-1)
        for_synthesis_iso = torch.cat((
            # data_mean[:,None],
            # data_std[:,None],
            P00_iso.log() if P00_iso.sum()!=0 else P00_iso, 
            S1_iso.log() if S1_iso.sum()!=0 else S1_iso, 
            C01_iso[:,select_and_index['select_2_iso']], 
#             C01_iso[:,select_and_index['select_2_iso']].imag, 
            C11_iso[:,select_and_index['select_3_iso']], 
#             C11_iso[:,select_and_index['select_3_iso']].imag
        ), dim=-1)
        for_synthesis_reduced = torch.cat((
            P00_reduced.log() if P00_reduced.sum()!=0 else P00_reduced,
            S1_reduced.log() if S1_reduced.sum()!=0 else S1_reduced,
            C01_reduced[:,select_and_index['select_2_iso']],
            C11_reduced[:,select_and_index['select_3_iso']],
        ), dim=-1)
        if normalization=='P11':
            for_synthesis     = torch.cat(
                (for_synthesis,     P11[:,select_and_index['select_2']].log()),         
                dim=-1)
            for_synthesis_iso = torch.cat(
                (for_synthesis_iso, P11_iso[:,select_and_index['select_2_iso']].log()), 
                dim=-1)
            
        return {'var': data.var((-2,-1)), 'mean': data.mean((-2,-1)),
                'P00':P00, 'P00_iso':P00_iso,
                'S1' : S1, 'S1_iso' : S1_iso,
                'C01':C01, 'C01_iso':C01_iso,
                'C11_pre_norm':C11_pre_norm, 'C11_pre_norm_iso':C11_pre_norm_iso,
                'C11': C11,'C11_iso': C11_iso,
                'P11':P11, 'P11_iso':P11_iso,
                'for_synthesis': for_synthesis, 'for_synthesis_iso': for_synthesis_iso,
                'for_synthesis_reduced': for_synthesis_reduced,
                'index_for_synthesis': index_for_synthesis,
                'index_for_synthesis_iso': index_for_synthesis_iso,
        }

    # ---------------------------------------------------------------------------
    #
    # Two-field cross scattering cov
    #
    # ---------------------------------------------------------------------------
    def scattering_cross_corr(
        self, data_a, data_b, if_large_batch=False, C11_criteria=None, 
        normalization='P00'
    ):
        '''
        Calculates the scattering correlations for a batch of images, including:
        orig. x orig.:     
                        P00 = <(I * psi)(I * psi)*> = L2(I * psi)^2
        orig. x modulus:   
                        C01 = <(I * psi2)(|I * psi1| * psi2)*> / factor
            when normalization == 'P00', factor = L2(I * psi2) * L2(I * psi1)
            when normalization == 'P11', factor = L2(I * psi2) * L2(|I * psi1| * psi2)
        modulus x modulus: 
                        C11_pre_norm = <(|I * psi1| * psi3)(|I * psi2| * psi3)>
                        C11 = C11_pre_norm / factor
            when normalization == 'P00', factor = L2(I * psi1) * L2(I * psi2)
            when normalization == 'P11', factor = L2(|I * psi1| * psi3) * L2(|I * psi2| * psi3)
        modulus x modulus (auto): 
                        P11 = <(|I * psi1| * psi2)(|I * psi1| * psi2)*>
        '''
        if C11_criteria is None: C11_criteria = 'j2>=j1'
            
        M, N, J, L = self.M, self.N, self.J, self.L
        N_image = data_a.shape[0]
        filters_set = self.filters_set
        weight = self.weight
                
        # convert numpy array input into torch tensors
        if type(data_a) == np.ndarray:
            data_a = torch.from_numpy(data_a)
        if type(data_b) == np.ndarray:
            data_b = torch.from_numpy(data_b)
            
        if self.device=='gpu':
            data_a = data_a.cuda()
            data_b = data_b.cuda()
        data_a = st.whiten(data_a)*weight
        data_b = st.whiten(data_b)*weight
        data_a_f = torch.fft.fftn(data_a, dim=(-2,-1))
        data_b_f = torch.fft.fftn(data_b, dim=(-2,-1))
        
        # initialize tensors for scattering coefficients
        P00_a = torch.zeros((N_image,J,L), dtype=data_a.dtype)
        P00_b = torch.zeros((N_image,J,L), dtype=data_a.dtype)
        P11_a = torch.zeros((N_image,J,J,L,L), dtype=data_a.dtype) + np.nan
        P11_b = torch.zeros((N_image,J,J,L,L), dtype=data_a.dtype) + np.nan
        C00 = torch.zeros((N_image,J,L), dtype=data_a_f.dtype)
        C11 = torch.zeros((N_image,4,J,J,J,L,L,L), dtype=data_a_f.dtype) + np.nan
        C11_sym = torch.zeros((N_image,J,J,L,L), dtype=data_a_f.dtype)
        
        C00_reduced = torch.zeros((N_image,J), dtype=data_a_f.dtype)
        C11_sym_reduced = torch.zeros((N_image,J,J), dtype=data_a_f.dtype)
        
        # move torch tensors to gpu device, if required
        if self.device=='gpu':
            P00_a     = P00_a.cuda()
            P00_b     = P00_b.cuda()
            P11_a     = P11_a.cuda()
            P11_b     = P11_b.cuda()
            C00       = C00.cuda()       
            C11       = C11.cuda()
            C11_sym   = C11_sym.cuda()
            C00_reduced   = C00_reduced.cuda()
            C11_sym_reduced = C11_sym_reduced.cuda()

        # calculate scattering fields
        I1_a = torch.fft.ifftn(
            data_a_f[:,None,None,:,:] * filters_set[None,:J,:,:,:], dim=(-2,-1)
        ).abs()
        I1_b = torch.fft.ifftn(
            data_b_f[:,None,None,:,:] * filters_set[None,:J,:,:,:], dim=(-2,-1)
        ).abs()
        I1_a_f = torch.fft.fftn(I1_a, dim=(-2,-1))
        I1_b_f = torch.fft.fftn(I1_b, dim=(-2,-1))
        
        P00_a = (I1_a**2).mean((-2,-1))
        P00_b = (I1_b**2).mean((-2,-1))
        
        C00 = (
            (data_a_f * torch.conj(data_b_f))[:,None,None,:,:] * filters_set[None,:J,:,:,:]**2
        ).mean((-2,-1)) /M/N
        Pa = P00_a; Pb = P00_b
        C00 = C00 / (Pa * Pb)**0.5
        
        # calculate the covariance and correlations of the scattering fields
        # only use the low-k Fourier coefs when calculating large-j scattering coefs.
        for j3 in range(0,J):
            dx3, dy3 = self.get_dxdy(j3)
            I1_a_f_small = cut_high_k_off(I1_a_f, dx3, dy3)
            I1_b_f_small = cut_high_k_off(I1_b_f, dx3, dy3)
            wavelet_f3 = cut_high_k_off(filters_set[j3], dx3, dy3)
            _, M3, N3 = wavelet_f3.shape
            wavelet_f3_squared = wavelet_f3**2
            # a normalization change due to the cutoff of frequency space
            fft_factor = 1 /(M3*N3) * (M3*N3/M/N)**2
            for j2 in range(0,j3+1):
                # [N_image,l2,l3,x,y]
                for j1 in range(0, j2+1):
                    if eval(C11_criteria):
                        if not if_large_batch:
                            # [N_image,l1,l2,l3,x,y]
                            C11[:,0,j1,j2,j3,:,:,:] = (
                                I1_a_f_small[:,j1].view(N_image,L,1,1,M3,N3) * 
                                torch.conj(I1_a_f_small[:,j2].view(N_image,1,L,1,M3,N3)) *
                                wavelet_f3_squared.view(1,1,1,L,M3,N3)
                            ).mean((-2,-1)) * fft_factor
                            C11[:,1,j1,j2,j3,:,:,:] = (
                                I1_b_f_small[:,j1].view(N_image,L,1,1,M3,N3) * 
                                torch.conj(I1_b_f_small[:,j2].view(N_image,1,L,1,M3,N3)) *
                                wavelet_f3_squared.view(1,1,1,L,M3,N3)
                            ).mean((-2,-1)) * fft_factor
                            C11[:,2,j1,j2,j3,:,:,:] = (
                                I1_a_f_small[:,j1].view(N_image,L,1,1,M3,N3) * 
                                torch.conj(I1_b_f_small[:,j2].view(N_image,1,L,1,M3,N3)) *
                                wavelet_f3_squared.view(1,1,1,L,M3,N3)
                            ).mean((-2,-1)) * fft_factor
                            C11[:,3,j1,j2,j3,:,:,:] = (
                                I1_b_f_small[:,j1].view(N_image,L,1,1,M3,N3) * 
                                torch.conj(I1_a_f_small[:,j2].view(N_image,1,L,1,M3,N3)) *
                                wavelet_f3_squared.view(1,1,1,L,M3,N3)
                            ).mean((-2,-1)) * fft_factor
                        else:
                            for l1 in range(L):
                            # [N_image,l2,l3,x,y]
                                C11[:,0,j1,j2,j3,l1,:,:] = (
                                    I1_a_f_small[:,j1,l1].view(N_image,1,1,M3,N3) * 
                                    torch.conj(I1_a_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                                    wavelet_f3_squared.view(1,1,L,M3,N3)
                                ).mean((-2,-1)) * fft_factor
                                C11[:,1,j1,j2,j3,l1,:,:] = (
                                    I1_b_f_small[:,j1,l1].view(N_image,1,1,M3,N3) * 
                                    torch.conj(I1_b_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                                    wavelet_f3_squared.view(1,1,L,M3,N3)
                                ).mean((-2,-1)) * fft_factor
                                C11[:,2,j1,j2,j3,l1,:,:] = (
                                    I1_a_f_small[:,j1,l1].view(N_image,1,1,M3,N3) * 
                                    torch.conj(I1_b_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                                    wavelet_f3_squared.view(1,1,L,M3,N3)
                                ).mean((-2,-1)) * fft_factor
                                C11[:,3,j1,j2,j3,l1,:,:] = (
                                    I1_a_f_small[:,j1,l1].view(N_image,1,1,M3,N3) * 
                                    torch.conj(I1_b_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                                    wavelet_f3_squared.view(1,1,L,M3,N3)
                                ).mean((-2,-1)) * fft_factor
        # define P11 from C11
        for j1 in range(J):
            for l1 in range(L):
                for j3 in range(j1, J):
                    P11_a[:,j1,j3,l1,:] = C11[:,0,j1,j1,j3,l1,l1,:].real
                    P11_b[:,j1,j3,l1,:] = C11[:,1,j1,j1,j3,l1,l1,:].real
        # normalizing C11
        if normalization=='P00':
            Pa = P00_a; Pb = P00_b
            #.view(N_image,J,1,1,L,1,1) *.view(N_image,1,J,1,1,L,1)
            C11[:,0] = C11[:,0] / (Pa[:,:,None,None,:,None,None] * Pa[:,None,:,None,None,:,None])**0.5
            C11[:,1] = C11[:,1] / (Pb[:,:,None,None,:,None,None] * Pb[:,None,:,None,None,:,None])**0.5
            C11[:,2] = C11[:,2] / (Pa[:,:,None,None,:,None,None] * Pb[:,None,:,None,None,:,None])**0.5
            C11[:,3] = C11[:,3] / (Pb[:,:,None,None,:,None,None] * Pa[:,None,:,None,None,:,None])**0.5
        if normalization=='P11':
            Pa = P11_a; Pb = P11_b
            #.view(N_image,J,1,J,L,1,L) * .view(N_image,1,1,J,L,J,L)
            C11[:,0] = C11[:,0] / (Pa[:,:,None,:,:,None,:] * Pa[:,None,:,:,None,:,:])**0.5
            C11[:,1] = C11[:,1] / (Pb[:,:,None,:,:,None,:] * Pb[:,None,:,:,None,:,:])**0.5
            C11[:,2] = C11[:,2] / (Pa[:,:,None,:,:,None,:] * Pb[:,None,:,:,None,:,:])**0.5
            C11[:,3] = C11[:,3] / (Pb[:,:,None,:,:,None,:] * Pa[:,None,:,:,None,:,:])**0.5
        for j1 in range(J):
            for l1 in range(L):
                C11_sym[:,j1,:,l1,:] += C11[:,2,j1,j1,:,l1,l1,:]
        
        # weighted average over angles to obtain reduced statistics
        weight_C00 = (P00_a * P00_b)**0.5
        weight_C11 = (P11_a * P11_b)**0.5  

        C00_reduced = torch.nansum(weight_C00*C00.real,-1) / torch.nansum(weight_C00, -1)
        C11_sym_reduced = torch.nansum(weight_C11*C11_sym.real,(-1,-2)) / torch.nansum(weight_C11,(-1,-2))
        C00_summary = torch.nansum((weight_C00*C00.real).reshape(N_image,-1), -1) / torch.nansum(weight_C00.reshape(N_image,-1),-1)
        C11_sym_summary = torch.nansum((weight_C11*C11_sym.real).reshape(N_image,-1), -1) / torch.nansum(weight_C11.reshape(N_image,-1), -1)

        weight_combined = torch.cat((weight_C00.reshape(N_image,-1), weight_C11.reshape(N_image,-1)), dim=-1)
        coeff_combined = torch.cat((C00.reshape(N_image,-1).real, C11_sym.reshape(N_image,-1).real), dim=-1)
        corr_combined = torch.nansum(weight_combined*coeff_combined, -1) / torch.nansum(weight_combined, -1)
            
        select_and_index = get_scattering_index(J, L, 'P00', C11_criteria)
        for_synthesis = torch.cat((
            C00.reshape((N_image, -1)).real.abs(),
            # C11_sym[:,select_and_index['select_2']].reshape((N_image, -1)).real,
        ), dim=-1)
        # for_synthesis_reduced = torch.cat((
        #     C00_reduced.reshape((N_image, -1)).real, 
        #     C11_sym_reduced[:,select_and_index['select_2_iso']].reshape((N_image, -1)).real, 
        # ), dim=-1)

        return {'corr':corr_combined, 'c00_summary': C00_summary, 'c11_summary': C11_sym_summary,
                'c00_reduced': C00_reduced, 'c11_reduced': C11_sym_reduced,
                'for_synthesis': for_synthesis}