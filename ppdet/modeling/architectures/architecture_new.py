from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
import numpy as np
from colorama import Fore, Style

from ppdet.utils.checkpoint import *

__all__ = ['ArchitectureNew']

def log(*args):
    print(f'{Fore.GREEN}Log: {Style.RESET_ALL}', *args)


class FastRCNNBlock(nn.Layer):
    def __init__(self, backbone, rpn_head, bbox_head):
        super(FastRCNNBlock, self).__init__()
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
    
    # Wrapped as an nn.Layer for loading weights, not for backpropogation
    def forward(self):
        return None


class FixedPatchPrompter_image(nn.Layer):
    def __init__(self, prompt_size):
        super(FixedPatchPrompter_image, self).__init__()
        self.psize = prompt_size
        self.patch = paddle.create_parameter(shape=[3, prompt_size, prompt_size], 
                                             dtype='float32', 
                                             default_initializer=nn.initializer.Normal(mean=0.0, std=1.0))
    
    def forward(self, inputs):
        image = inputs['image']
        prompt = paddle.zeros(shape=image.shape)
        prompt[:, :, :self.psize, :self.psize] = self.patch
        inputs['image'] = image + prompt
        return inputs
    

@register
class ArchitectureNew(BaseArch):
    __category__ = 'architecture'

    def __init__(self,
                 prompter_patch_size,
                 backbone_wanted,
                 backbone_supervised,
                 rpn_head_wanted,
                 rpn_head_supervised,
                 bbox_head_wanted,
                 bbox_head_supervised,
                 bbox_post_process,
                 weights_wanted_url,
                 weights_supervised_url,
                 neck=None):
        super(ArchitectureNew, self).__init__()
        self.neck = neck
        self.prompter = FixedPatchPrompter_image(prompt_size=prompter_patch_size)
        self.fasterRCNNs = {'wanted': FastRCNNBlock(backbone_wanted, 
                                                    rpn_head_wanted, 
                                                    bbox_head_wanted), 
                            'supervised': FastRCNNBlock(backbone_supervised, 
                                                    rpn_head_supervised, 
                                                    bbox_head_supervised)}
        self.bbox_post_process = bbox_post_process
        load_pretrain_weight(self.fasterRCNNs['wanted'], weights_wanted_url)
        load_pretrain_weight(self.fasterRCNNs['supervised'], weights_supervised_url)

        for name, param in self.named_parameters():
            log('Optimized paramter:', name, 'stop_gradients', param.stop_gradient )
        log('Dynamic Graph Mode', paddle.in_dynamic_mode())
        

    def init_cot_head(self, relationship):
        self.bbox_head.init_cot_head(relationship)

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone_wanted = create(cfg['backbone_wanted'], **kwargs)
        backbone_supervised = create(cfg['backbone_supervised'], **kwargs)
        kwargs = {'input_shape': backbone_wanted.out_shape}
        rpn_head_wanted = create(cfg['rpn_head_wanted'], **kwargs)
        rpn_head_supervised = create(cfg['rpn_head_supervised'], **kwargs)
        bbox_head_wanted = create(cfg['bbox_head_wanted'], **kwargs)
        bbox_head_supervised = create(cfg['bbox_head_supervised'], **kwargs)
        bbox_post_process = create(cfg['bbox_post_process'], **kwargs)
        return {
            'backbone_wanted': backbone_wanted,
            'backbone_supervised': backbone_supervised,
            'rpn_head_wanted': rpn_head_wanted,
            'rpn_head_supervised': rpn_head_supervised,
            'bbox_head_wanted': bbox_head_wanted,
            'bbox_head_supervised': bbox_head_supervised,
            'bbox_post_process': bbox_post_process,
        }

    def _predict(self, fasterRCNN):
        if not isinstance(fasterRCNN, FastRCNNBlock):
            raise ValueError(f'{fasterRCNN} is not a FastRCNNBlock instance.')
        self.inputs = self.prompter(self.inputs)
        body_feats = fasterRCNN.backbone(self.inputs)
        rois, rois_num, _ = fasterRCNN.rpn_head(body_feats, self.inputs)
        preds, _ = fasterRCNN.bbox_head(body_feats, rois, rois_num, None)
        im_shape = self.inputs['im_shape']
        scale_factor = self.inputs['scale_factor']
        bbox, bbox_num, nms_keep_idx = self.bbox_post_process(preds, (rois, rois_num), im_shape, scale_factor)
        # rescale the prediction back to origin image
        bboxes, bbox_pred, bbox_num = self.bbox_post_process.get_pred(bbox, bbox_num, im_shape, scale_factor)
        return bbox_pred, bbox_num, body_feats

    def _forward(self):
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        if self.training:
            with paddle.no_grad():
                self.fasterRCNNs['supervised'].eval()
                bbox_pred, bbox_num, body_feats_supervised = self._predict(self.fasterRCNNs['supervised'])
                choose = bbox_pred[:, 1]
                gt_class = bbox_pred[:, :1]
                gt_bbox = bbox_pred[:, 2:]
                # threshold
                gt_class = gt_class[choose > 0.5]
                gt_bbox = gt_bbox[choose > 0.5]
            self.fasterRCNNs['supervised'].train()
            self.inputs['gt_class'] = (gt_class,)
            self.inputs['gt_bbox'] = (gt_bbox,)
            body_feats_wanted = self.fasterRCNNs['wanted'].backbone(self.inputs)
            rois, rois_num, rpn_loss = self.fasterRCNNs['wanted'].rpn_head(body_feats_wanted, self.inputs)  # USE GT
            bbox_loss, _ = self.fasterRCNNs['wanted'].bbox_head(body_feats_wanted, rois, rois_num, self.inputs)  # USE GT
            feat_loss = F.l1_loss(body_feats_wanted[0], body_feats_supervised[0])
            return rpn_loss, bbox_loss, feat_loss
        else:
            self.fasterRCNNs['wanted'].eval()
            bbox_pred, bbox_num, _ = self._predict(self.fasterRCNNs['wanted'])
            return bbox_pred, bbox_num

    def get_loss(self, ):
        rpn_loss, bbox_loss, feat_loss = self._forward()
        loss = {}
        loss.update(rpn_loss)
        loss.update(bbox_loss)
        total_loss = paddle.add_n(list(loss.values()))
        total_loss += feat_loss
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        if self.use_extra_data:
            bbox_pred, bbox_num, extra_data = self._forward()
            output = {
                'bbox': bbox_pred,
                'bbox_num': bbox_num,
                'extra_data': extra_data
            }
        else:
            bbox_pred, bbox_num = self._forward()
            output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
        return output

    def target_bbox_forward(self, data):
        body_feats = self.backbone(data)
        if self.neck is not None:
            body_feats = self.neck(body_feats)
        rois = [roi for roi in data['gt_bbox']]
        rois_num = paddle.concat([paddle.shape(roi)[0:1] for roi in rois])

        preds, _ = self.bbox_head(body_feats, rois, rois_num, None, cot=True)
        return preds

    def relationship_learning(self, loader, num_classes_novel):
        print('computing relationship')
        train_labels_list = []
        label_list = []

        for step_id, data in enumerate(loader):
            _, bbox_prob = self.target_bbox_forward(data)
            batch_size = data['im_id'].shape[0]
            for i in range(batch_size):
                num_bbox = data['gt_class'][i].shape[0]
                train_labels = data['gt_class'][i]
                train_labels_list.append(train_labels.numpy().squeeze(1))
            base_labels = bbox_prob.detach().numpy()[:, :-1]
            label_list.append(base_labels)

        labels = np.concatenate(train_labels_list, 0)
        probabilities = np.concatenate(label_list, 0)
        N_t = np.max(labels) + 1
        conditional = []
        for i in range(N_t):
            this_class = probabilities[labels == i]
            average = np.mean(this_class, axis=0, keepdims=True)
            conditional.append(average)
        return np.concatenate(conditional)
