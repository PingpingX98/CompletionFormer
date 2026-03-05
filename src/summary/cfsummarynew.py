from PIL import Image
from . import BaseSummary
import imageio
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm


from tqdm import tqdm

import cv2

""" cmap = 'jet'
cm = plt.get_cmap(cmap) """

class CompletionFormerSummarynew(BaseSummary):
    def __init__(self, log_dir, mode, args, loss_name, metric_name):
        assert mode in ['train', 'val', 'test'], \
            "mode should be one of ['train', 'val', 'test'] " \
            "but got {}".format(mode)

        super(CompletionFormerSummarynew, self).__init__(log_dir, mode, args)

        self.log_dir = log_dir
        self.mode = mode
        self.args = args

        self.loss = []
        self.metric = []

        self.loss_name = loss_name
        self.metric_name = metric_name

        self.path_output = None

        # ImageNet normalization
        self.img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def update(self, global_step, sample, output):
        if self.loss_name is not None:
            self.loss = np.concatenate(self.loss, axis=0)
            self.loss = np.mean(self.loss, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Loss')]
            for idx, loss_type in enumerate(self.loss_name):
                val = self.loss[0, idx]
                self.add_scalar('Loss/' + loss_type, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(loss_type, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_loss = open(self.f_loss, 'a')
            f_loss.write('{:05d} | {}\n'.format(global_step, msg))
            f_loss.close()

        if self.metric_name is not None:
            self.metric = np.concatenate(self.metric, axis=0)
            self.metric = np.mean(self.metric, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Metric')]
            for idx, name in enumerate(self.metric_name):
                val = self.metric[0, idx]
                self.add_scalar('Metric/' + name, val, global_step)

                msg += ["{:<s}: {:.5f}  ".format(name, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_metric = open(self.f_metric, 'a')
            f_metric.write('{:05d} | {}\n'.format(global_step, msg))
            f_metric.close()

        # Un-normalization
        rgb = sample['rgb'].detach()
        rgb.mul_(self.img_std.type_as(rgb)).add_(self.img_mean.type_as(rgb))
        rgb = rgb.data.cpu().numpy()

        dep = sample['dep'].detach().data.cpu().numpy()
        gt = sample['gt'].detach().data.cpu().numpy()
        pred = output['pred'].detach().data.cpu().numpy()

        """ if output['confidence'] is not None:
            confidence = output['confidence']
        else:
            confidence = np.zeros_like(dep) """
        '''
        preds = [d.detach().data.cpu().numpy() for d in output['pred_inter']]
        if output['confidence'] is not None:
            confidence = output['confidence']
            if isinstance(confidence, (list, tuple)):
                confidence = [c.data.cpu().numpy() for c in confidence]
                if len(confidence) == 1:    # 意味着模型对所有预测结果的置信度都一样
                    confidence = confidence * len(preds)    # 复制confidence，使其和preds的预测结果数一样
            else:
                confidence = confidence.data.cpu().numpy()
                confidence = [confidence, ] * len(preds)

        else:
            confidence = [np.zeros_like(dep)] * len(preds)
        '''    
        
        num_summary = rgb.shape[0]
        if num_summary > self.args.num_summary:
            num_summary = self.args.num_summary

            rgb = rgb[0:num_summary, :, :, :]
            dep = dep[0:num_summary, :, :, :]
            gt = gt[0:num_summary, :, :, :]
            pred = pred[0:num_summary, :, :, :]
            # confidence = confidence[0:num_summary, :, :, :]

        rgb = np.clip(rgb, a_min=0, a_max=1.0)
        dep = np.clip(dep, a_min=0, a_max=self.args.max_depth)
        gt = np.clip(gt, a_min=0, a_max=self.args.max_depth)
        pred = np.clip(pred, a_min=0, a_max=self.args.max_depth)
        # confidence = np.clip(confidence, a_min=0, a_max=1.0)

        list_img = []

        for b in range(0, num_summary):
            rgb_tmp = rgb[b, :, :, :]
            dep_tmp = dep[b, 0, :, :]
            gt_tmp = gt[b, 0, :, :]
            pred_tmp = pred[b, 0, :, :] 
            # confidence_tmp = confidence[b, 0, :, :]
            
            # norm = plt.Normalize(vmin=gt_tmp.min(), vmax=gt_tmp.max())

            dep_tmp = 255.0 * dep_tmp / self.args.max_depth
            gt_tmp = 255.0 * gt_tmp / self.args.max_depth
            pred_tmp = 255.0 * pred_tmp / self.args.max_depth
            # confidence_tmp = 255.0 * confidence_tmp
            cmap = cm.jet
            
            # 输入应该是归一化到 [0, 1] 之间的浮点数
            dep_tmp = (dep_tmp.astype('float32') / 255.0)
            gt_tmp = (gt_tmp.astype('float32') / 255.0)
            pred_tmp = (pred_tmp.astype('float32') / 255.0)

            # 应用 colormap（返回 RGBA），取 RGB 部分并转换成 uint8
            dep_tmp = (cmap(dep_tmp)[..., :3] * 255).astype('uint8')
            gt_tmp = (cmap(gt_tmp)[..., :3] * 255).astype('uint8')
            pred_tmp = (cmap(pred_tmp)[..., :3] * 255).astype('uint8')
            """ dep_tmp = cm(dep_tmp.astype('uint8'))
            gt_tmp = cm(gt_tmp.astype('uint8'))
            pred_tmp = cm(pred_tmp.astype('uint8')) """
            # confidence_tmp = cm(confidence_tmp.astype('uint8'))

            dep_tmp = np.transpose(dep_tmp[:, :, :3], (2, 0, 1))
            gt_tmp = np.transpose(gt_tmp[:, :, :3], (2, 0, 1))
            pred_tmp = np.transpose(pred_tmp[:, :, :3], (2, 0, 1))
            # confidence_tmp = np.transpose(confidence_tmp[:, :, :3], (2, 0, 1))

            img = np.concatenate((rgb_tmp, dep_tmp, pred_tmp, gt_tmp,), axis=1)

            list_img.append(img)

        img_total = np.concatenate(list_img, axis=2)
        img_total = torch.from_numpy(img_total)

        self.add_image(self.mode + '/images', img_total, global_step)

        self.add_scalar('Etc/gamma', output['gamma'], global_step)

        self.flush()

        # Reset
        self.loss = []
        self.metric = []

    def save(self, epoch, idx, sample, output):
        with torch.no_grad():
            if self.args.save_result_only:
                # print("Saving only result...")
                # self.path_output = '{}/{}/epoch{:04d}'.format(self.log_dir,
                #                                               self.mode, epoch)
                self.path_output = self.log_dir
                dep = sample['dep'].detach()
                
                os.makedirs(self.path_output, exist_ok=True)
                path_save_pred = os.path.join(self.path_output, 'depth')
                path_save_color = os.path.join(self.path_output, 'depthcolor')
                os.makedirs(path_save_pred, exist_ok=True)
                os.makedirs(path_save_color, exist_ok=True)
                path_save_pred = '{}/{:08d}.png'.format(path_save_pred, idx)
                path_save_color = '{}/{:08d}.png'.format(path_save_color, idx)
                
                pred = output['pred'].detach()

                pred = torch.clamp(pred, min=0)

                pred = pred[0, 0, :, :].data.cpu().numpy()
                print(f"pred shape: {pred.shape}")

                pred =  1 + (pred - pred.min()) / (pred.max() - pred.min())
                pred_test = pred[:, 30:-20]
                print(f"pred_test shape: {pred_test.shape}")
                print(f"pred_test min is {pred_test.min()}, max is {pred_test.max()}")
                print(f"pred min is {pred.min()}, max is {pred.max()}")
                pred = (pred*256.0).astype(np.uint16)
                color_depth = self.Colorize(pred, min_distance=pred[pred > 0].min(), max_distance=pred.max())
                imageio.imwrite(path_save_pred, pred)
                imageio.imwrite(path_save_color, color_depth)
                 
            else:
                cmap = cm.jet
                rgb = sample['rgb'].detach()
                dep = sample['dep'].detach()
                pred = output['pred'].detach()
                gt = sample['gt'].detach()
                file_name = sample['file_name'][0]
                pred = torch.clamp(pred, min=0)

                # Un-normalization
                rgb.mul_(self.img_std.type_as(rgb)).add_(
                    self.img_mean.type_as(rgb))

                rgb = rgb[0, :, :, :].data.cpu().numpy()
                dep = dep[0, 0, :, :].data.cpu().numpy()
                pred = pred[0, 0, :, :].data.cpu().numpy()
                gt = gt[0, 0, :, :].data.cpu().numpy()
                
                norm = plt.Normalize(vmin=gt.min(), vmax=gt.max())
                rgb = np.transpose(rgb, (1, 2, 0))

                self.path_output = '{}/{}/epoch{:04d}/{:08d}'.format(
                    self.log_dir, self.mode, epoch, idx)
                self.path_output_selection = '{}/{}/val_selection_cropped/velodyne_raw'.format(
                    self.log_dir, self.mode, epoch)
                os.makedirs(self.path_output, exist_ok=True)
                os.makedirs(self.path_output_selection, exist_ok=True)

                path_save_rgb = '{}/01_rgb.png'.format(self.path_output)
                # path_save_dep = '{}/02_dep.png'.format(self.path_output)
                path_save_dep = '{}/{}'.format(self.path_output_selection, file_name)
                path_save_pred = '{}/03_pred.png'.format(self.path_output)
                path_save_pred_visual = '{}/04_pred_final.png'.format(
                    self.path_output)  
                #rgb = rgb.astype(np.uint16)
                pred_uint16 = (pred * 256).astype(np.uint16)
                depth_color = self.colorize(pred, gt)
                #imageio.imwrite(path_save_rgb, rgb
                plt.imsave(path_save_rgb, rgb, cmap=cmap)
                imageio.imwrite(path_save_dep, (dep * 256).astype(np.uint16))
                imageio.imwrite(path_save_pred, pred_uint16)
                plt.imsave(path_save_pred_visual, depth_color)
                
                '''
                plt.imshow(pred, cmap='gray', vmin=0, vmax=10)
                plt.colorbar()
                plt.show()
                '''
    def ColorizeNew(self, depth,norm_type='LogNorm', offset=1.):
        cmap = cm.jet
        depth = (depth - depth.min()) / (depth.max() - depth.min()) + offset
        Norm = getattr(colors, norm_type)
        norm = Norm(vmin=depth.min(), vmax=depth.max(), clip=True)
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        depth_color = (255 * m.to_rgba(depth)[:, :, 0:3]).astype(np.uint8)
        return depth_color    
    
    def colorize(self, depth, gt):
        cmap = cm.jet
        vmin = np.percentile(gt, 1)
        vmax = gt.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        depth_color =(255 * m.to_rgba(depth)[:, :, 0:3]).astype(np.uint8)
        return depth_color
    import os


    def Colorize(self, depth, min_distance=None, max_distance=None, radius=None, norm_type='LogNorm', cmap=cm.jet):
        # norm = colors.Normalize(vmin=min_distance, vmax=max_distance)
        # norm = colors.Normalize(vmin=min_distance, vmax=max_distance,clip=True)
        Norm = getattr(colors, norm_type)
        if min_distance == max_distance:
            min_distance = 1e-5
        if norm_type == 'PowerNorm':
            norm = Norm(vmin=min_distance, vmax=max_distance, clip=True, gamma=0.5)
        else:
            norm = Norm(vmin=min_distance, vmax=max_distance, clip=True)
        # norm = colors.LogNorm(vmin=min_distance, vmax=max_distance,clip=False)
        # norm = colors.PowerNorm(gamma=1. / 2.),
        cmap = cmap
        # cmap = cm.gray
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        if radius == None:
            depth_color = (255 * m.to_rgba(depth)[:, :, 0:3]).astype(np.uint8)
            # depth_color[depth <= 0] = [0, 0, 0]
            # depth_color[np.isnan(depth)] = [0, 0, 0]
            # depth_color[depth == np.inf] = [0, 0, 0]
        else:
            pos = np.argwhere(depth > 0)
            depth_color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
            for i in range(pos.shape[0]):
                color = tuple([int(255 * value) for value in m.to_rgba(depth[pos[i, 0], pos[i, 1]])[0:3]])
                cv2.circle(depth_color, (pos[i, 1], pos[i, 0]), radius, (color[0], color[1], color[2]), -1)
        return depth_color


    def Colorize_Transparent(depth, min_distance=None, max_distance=None, radius=None, norm_type='LogNorm', cmap=cm.jet):
        Norm = getattr(colors, norm_type)

        if min_distance == max_distance:
            min_distance = 1e-5

        if norm_type == 'PowerNorm':
            norm = Norm(vmin=min_distance, vmax=max_distance, clip=True, gamma=0.5)
        else:
            norm = Norm(vmin=min_distance, vmax=max_distance, clip=True)

        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        if radius is None:
            rgba_image = (255 * m.to_rgba(depth)).astype(np.uint8)


            mask = (depth <= 0) | (np.isnan(depth)) | (np.isinf(depth))
            rgba_image[mask, 3] = 0
            depth_color = rgba_image
        else:
            pos = np.argwhere(depth > 0)

            depth_color = np.zeros((depth.shape[0], depth.shape[1], 4), dtype=np.uint8)

            for i in range(pos.shape[0]):
                rgba = m.to_rgba(depth[pos[i, 0], pos[i, 1]])
                color = tuple([int(255 * value) for value in rgba])

                cv2.circle(depth_color, (pos[i, 1], pos[i, 0]), radius, color, -1)

        return depth_color        
                
                
                
                
                