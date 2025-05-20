from net import *
from net.losses import reconstruction_loss, illumination_smooth_loss, reflectance_smooth_loss
from utils.image_io import *
from utils.init import _init_all
from skimage.measure import compare_psnr
from PIL import ImageFilter
from cv2.ximgproc import guidedFilter
import os
import cv2
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class Segmentation(object):
    
    def __init__(self, image_name, image, plot_during_training=True,
                 first_step_iter_num=2500,
                 second_step_iter_num=1500,
                 bg_hint=None, fg_hint=None, light_hint=None, output_dir=None,
                 downsampling_factor=0.1, downsampling_number=0):
        self.image = image
        if bg_hint is None or fg_hint is None or light_hint is None: 
            raise Exception("Hints must be provided")
        self.image_name = image_name
        self.input_depth = 2
        self.plot_during_training = plot_during_training
        self.downsampling_factor = downsampling_factor
        self.downsampling_number = downsampling_number
        self.kernel_net = None
        self.show_every = 500
        self.bg_hint = bg_hint
        self.fg_hint = fg_hint
        self.light_hint = light_hint
        self.left_net = None
        self.right_net = None
        self.RRD_net = None
        self.images = None
        self.images_torch = None
        self.left_net_inputs = None
        self.right_net_inputs = None
        self.kernel_net_inputs = None
        self.left_net_outputs = None
        self.right_net_outputs = None
        self.second_step_done = False
        self.kernel_net_outputs = None
        self.mask_fg = None
        self.mask_light = None
        self.parameters = None
        self.parametersRRD = None
        self.fixed_masks = None
        self.first_step_iter_num = first_step_iter_num
        self.second_step_iter_num = second_step_iter_num
        self.total_loss = None
        self.current_gradient = None
        self.current_result = None
        self.learning_rate = 0.001
        self._init_all = _init_all(self)
        self.gamma = 0.85
        self.weight = 1.0

    def _triple_convolution(self, input_tensor, kernel):
        _, _, h, w = input_tensor.shape

        kernel_size = kernel.shape[-1]
        pad_size = kernel_size // 2

        def conv_block(x, k):
            x_pad = F.pad(x, (pad_size,)*4, mode='replicate')
            x_conv = F.conv2d(x_pad, k, padding=0, groups=3)
            x_blur = F.conv2d(x_conv, self.gauss_kernel, 
                             padding=self.gauss_kernel.shape[-1]//2, 
                             groups=3)
            return x_blur

        if not hasattr(self, 'gauss_kernel'):
            gauss = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32)
            self.gauss_kernel = (gauss/gauss.sum()).view(1,1,3,3).repeat(3,1,1,1)
            self.gauss_kernel = self.gauss_kernel.to(input_tensor.device)

        conv1 = F.leaky_relu(conv_block(input_tensor, kernel), 0.01)
        conv2 = F.leaky_relu(conv_block(conv1, kernel), 0.01)
        conv3 = conv_block(conv2, kernel)

        return conv3
    
    def optimize_step1(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # step 1
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        print('----------------------First_step-----------------------')
        for j in range(self.first_step_iter_num):
            optimizer.zero_grad()
            self._initialize_any_step(j)
            self._step1_optimize_with_hints(j)
            self._finalize_iteration()
            if self.second_step_done:
                break
            if self.plot_during_training:
                self._iteration_plot_closure(j)
            optimizer.step()
        self._update_result_closure()

        glow_map = cv2.imread(output_dir + '/glow_map.jpg')
        height = glow_map.shape[0]
        width = glow_map.shape[1]
        gray_map = np.zeros((height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                gray_map[i,j] = max(glow_map[i,j][0], glow_map[i,j][1], glow_map[i,j][2])
        gray_map = (gray_map - np.min(gray_map)) / (np.max(gray_map) - np.min(gray_map) + 1e-6)
        gray_map = cv2.GaussianBlur(gray_map, (5, 5), sigmaX=10.5)
        cv2.imwrite(output_dir + '/gray_map.jpg', (gray_map * 255).astype(np.uint8))
        

    def optimize_step2(self, transission, weight_map):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        transission = np_to_torch(transission).to('cuda')
        weight_map = np_to_torch(weight_map/255.0).to('cuda')
        weight_map = weight_map.to(dtype = torch.float32)
        optimizer = torch.optim.Adam(self.parametersRRD, lr=self.learning_rate)
        
        print('----------------------Second_step-----------------------')
        for j in range(self.second_step_iter_num):
            optimizer.zero_grad()
            self._initialize_any_step(j)
            self._step2_optimize_with_hints(j, transission)
            optimizer.step()
            
            stage = (j+1)// 500
            if 1 <= stage <= 3:
                img_path = f'/enhanced_{(stage*500)-1}.jpg'
                enhance, _, _ = prepare_image(output_dir + img_path)
                enhance = np_to_torch(enhance).to('cuda')
                transission = transission * weight_map + enhance * (1 - weight_map)
                transission = transission.to(dtype=torch.float32)

    def finalize(self):
        print('finalize')
        save_image("original", self.images[0], output_dir+'/')

    def _update_result_closure(self):
        self._finalize_iteration()
        self._fix_mask()

    def _fix_mask(self):
        mask_np = torch_to_np(self.mask_fg)
        new_mask_np = [np.array([guidedFilter(self.images[0].transpose(1, 2, 0).astype(np.float32),
                                               mask_np[0].astype(np.float32), 50, 1e-4)])]
        def to_bin(x):
            v = np.zeros_like(x)
            v[x > 0.5] = 1
            return v
        self.fixed_masks = [to_bin(m) for m in new_mask_np]

    def _initialize_any_step(self, iteration):
        if iteration == self.first_step_iter_num - 1:
            reg_noise_std = 0
        elif iteration < 1000:
            reg_noise_std = (1 / 1000.) * (iteration // 100)
        else:
            reg_noise_std = 1 / 1000.
        right_net_inputs = []
        left_net_inputs = []
        kernel_net_inputs = []

        # creates inputs by adding small noise
        for left_net_original_input, right_net_original_input, kernel_net_original_input \
                in zip(self.left_net_inputs, self.right_net_inputs, self.kernel_net_inputs):
            left_net_inputs.append(
                left_net_original_input + (left_net_original_input.clone().normal_() * reg_noise_std))          
            right_net_inputs.append(
                right_net_original_input + (right_net_original_input.clone().normal_() * reg_noise_std))
            kernel_net_inputs.append(
                kernel_net_original_input + (kernel_net_original_input.clone().normal_() * reg_noise_std))
        self.left_net_outputs = [self.left_net(left_net_input) for left_net_input in left_net_inputs]
        self.right_net_outputs = [self.right_net(right_net_input) for right_net_input in right_net_inputs]       
        self.kernel_net_outputs = [self.kernel_net(kernel_net_input) for kernel_net_input in kernel_net_inputs]
        for kernel_net_output in self.kernel_net_outputs:
            kernel_net_output = kernel_net_output.view(-1, 1, 15, 15)
            kernel_net_output = kernel_net_output.repeat(3, 1, 1, 1)

        self.mask_fg = np_to_torch(self.fg_hint).type(torch.cuda.FloatTensor)
        self.mask_light = np_to_torch(self.light_hint).type(torch.cuda.FloatTensor)
        self.total_loss = 0

    def _step1_optimize_with_hints(self, iteration):        
        if iteration <= 2000:
            self.total_loss += sum(self.extended_l1_loss(left_net_output, image_torch, fg_hint) for
                                   left_net_output, fg_hint, image_torch
                                   in zip(self.left_net_outputs, self.fg_hints_torch, self.images_torch))
            self.total_loss += sum(self.extended_l1_loss(right_net_output, image_torch, bg_hint) for
                                   right_net_output, bg_hint, image_torch
                                   in zip(self.right_net_outputs, self.bg_hints_torch, self.images_torch))
                
        if iteration > 500:
            mask_out = self.mask_fg
            mask_light = self.mask_light
            for left_out, right_out, kernel_out, original_image_torch in zip(self.left_net_outputs,
                                                                        self.right_net_outputs,
                                                                        self.kernel_net_outputs,
                                                                        self.images_torch):

                kernel_out = kernel_out.view(-1, 1, 15, 15).repeat(3, 1, 1, 1)
                GMap = self._triple_convolution(left_out, kernel_out)
                
                self.total_loss += 10 * self.per_loss(GMap * (1 - mask_light) + right_out, original_image_torch) 
                self.total_loss += 0.5 * self.exclusion_loss(right_out, GMap)
                self.total_loss += 300 * self.TV_loss(GMap)
                self.current_gradient = self.gray_loss(mask_out)

            iteration = min(iteration, 1000)
            self.total_loss += (0.001 * (iteration // 100)) * self.current_gradient

        self.total_loss.backward(retain_graph=True)

    def _step2_optimize_with_hints(self, iteration, trans): 
        transission = trans
        illumination, reflectance = self.RRD_net(transission)
        loss_recons = 2e-5 * reconstruction_loss(transission, illumination, reflectance)
        loss_illu = 2e-5 * illumination_smooth_loss(transission, illumination)
        loss_reflect = 2e-5 * reflectance_smooth_loss(transission, illumination, reflectance)
        loss_RRD = loss_recons + loss_reflect + loss_illu
        self.total_loss = self.total_loss + loss_RRD

        adjust_illu = torch.pow(illumination, self.gamma)
        res_image = adjust_illu*(transission/illumination)
        res_image = torch.clamp(res_image, min=0, max=1)
        res_img = transforms.ToPILImage()(res_image.squeeze(0))

        if iteration % self.show_every == self.show_every - 1:
            res_img.save(output_dir + '/enhanced_{}'.format(iteration) + '.jpg')

        if (iteration+1)%200 == 0:
            print("iter:", iteration, '  RRD loss:', float(loss_RRD.data))
            
        self.total_loss.backward(retain_graph=True)       

    def _finalize_iteration(self):
        mask_light_np = torch_to_np(self.mask_light)
        right_out_np = torch_to_np(self.right_net_outputs[0])
        kernel_out = self.kernel_net_outputs[0]
        kernel_out = kernel_out.view(-1, 1, 15, 15).repeat(3, 1, 1, 1)
        original_image = self.images[0]
        GMap = self._triple_convolution(self.left_net_outputs[0], kernel_out)
        
        GMap_out_np = torch_to_np(GMap)
        self.current_psnr = compare_psnr(original_image, GMap_out_np * (1 - mask_light_np)+right_out_np)

    def _iteration_plot_closure(self, iter_number):            
        if self.current_gradient is not None:
            print('Iter {:5d} total_loss {:5f} grad {:5f} PSNR {:5f} '.format(iter_number,self.total_loss.item(),self.current_gradient.item(),self.current_psnr),'\r', end='')
        else:
            print('Iter {:5d} total_loss {:5f} PSNR {:5f} '.format(iter_number, self.total_loss.item(),self.current_psnr),'\r', end='')
        if iter_number % self.show_every == self.show_every - 1:
            self._plot_with_name()

    def _plot_with_name(self):
        mask_light = self.mask_light
        for i, (left_out, kernel_out, image) in enumerate(zip(self.left_net_outputs,
                                                                       self.kernel_net_outputs,
                                                                        self.images)):
            mask_light_np = torch_to_np(mask_light)
            kernel_out = kernel_out.view(-1, 1, 15, 15).repeat(3, 1, 1, 1)
            GMap = self._triple_convolution(left_out, kernel_out)

            GMap_out_np = torch_to_np(GMap)
            save_image("glow_map", GMap_out_np, output_dir+'/')
            save_image("noGlow", image - GMap_out_np * (1 - mask_light_np) * self.weight, output_dir+'/')
                                                                                                             
                                     
if __name__ == "__main__":
    img = glob.glob(r"images/*")
    for filename in img:
        i, _, _ = prepare_image(filename)
        name = os.path.basename(filename)
        imgname = os.path.basename(os.path.splitext(filename)[0])
        print(imgname)
        output_dir = 'Results/'+imgname
        os.makedirs(output_dir, exist_ok=True)
        bg, _, _ = prepare_image('output_bg/' + name)
        fg, _, _ = prepare_image('output_fg/' + name)
        light, _, _ = prepare_image('output_light/' + name)
        t = Segmentation(imgname, i, bg_hint=bg, fg_hint=fg, light_hint=light, output_dir=output_dir)   
        t.optimize_step1()
        t.finalize()
        transmission, _, _ = prepare_image(output_dir + '/noGlow.jpg')
        weight_map = cv2.imread(output_dir + '/gray_map.jpg')
        weight_map = cv2.cvtColor(weight_map,cv2.COLOR_RGB2GRAY)
        t.optimize_step2(transmission, weight_map)