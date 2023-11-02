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

def log_value(*args):
    print(f'{Fore.GREEN}Log Value: {Style.RESET_ALL}', *args)


class FastRCNNBlock(nn.Layer):
    def __init__(self, backbone, rpn_head, bbox_head, bbox_post_process):
        super(FastRCNNBlock, self).__init__()
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.bbox_post_process = bbox_post_process
    
    def forward(self):
        return None

def _strip_postfix(path):
    path, ext = os.path.splitext(path)
    assert ext in ['', '.pdparams', '.pdopt', '.pdmodel'], \
            "Unknown postfix {} from weights".format(ext)
    return path



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
    
    # @classmethod
    # def from_config(cls, cfg, *args, **kwargs):
    #     prompt_size = cfg['prompt_size']
    #     return {'prompt_size': prompt_size}


@register
class ArchitectureNew(BaseArch):
    __category__ = 'architecture'

    @staticmethod
    def _get_pretrained_weights(pretrain_weight):
        if is_url(pretrain_weight):
            pretrain_weight = get_weights_path(pretrain_weight)
        path = _strip_postfix(pretrain_weight)
        if not (os.path.isdir(path) or os.path.isfile(path) or
                os.path.exists(path + '.pdparams')):
            raise ValueError("Model pretrain path `{}` does not exists. "
                            "If you don't want to load pretrain model, "
                            "please delete `pretrain_weights` field in "
                            "config file.".format(path))
        weights_path = path + '.pdparams'
        param_state_dict = paddle.load(weights_path)
        return param_state_dict
    
    @staticmethod
    def _prepare_weights(weights_wanted_url, weights_supervised_url):
        weights_wanted = ArchitectureNew._get_pretrained_weights(weights_wanted_url)
        weights_supervised = ArchitectureNew._get_pretrained_weights(weights_supervised_url)

        def add_sufix(string, sufix):
            tokens = string.split('.')
            tokens[0] = tokens[0] + sufix
            return '.'.join(tokens)
        
        weights_wanted = {add_sufix(key, '_wanted'): value for key, value in weights_wanted.items()}
        weights_supervised = {add_sufix(key, '_supervised'): value for key, value in weights_supervised.items()}
        weights_wanted.update(weights_supervised)
        paddle.save(weights_wanted, os.path.join('output', 'customed', 'model.pdparams'))


    def __init__(self,
                 prompter_patch_size,
                 backbone_wanted,
                 backbone_supervised,
                 rpn_head_wanted,
                 rpn_head_supervised,
                 bbox_head_wanted,
                 bbox_head_supervised,
                 bbox_post_process_wanted,
                 bbox_post_process_supervised,
                 weights_wanted_url,
                 weights_supervised_url,
                 neck=None):
        super(ArchitectureNew, self).__init__()
        self.neck = neck

        self.prompter = FixedPatchPrompter_image(prompt_size=prompter_patch_size)
        # log_value(type(self.prompter))

        self.backbone_wanted = backbone_wanted
        self.rpn_head_wanted = rpn_head_wanted
        self.bbox_head_wanted = bbox_head_wanted
        self.bbox_post_process_wanted = bbox_post_process_wanted
        self.backbone_supervised = backbone_supervised
        self.rpn_head_supervised = rpn_head_supervised
        self.bbox_head_supervised = bbox_head_supervised
        self.bbox_post_process_supervised = bbox_post_process_supervised

        self.fasterRCNN_wanted = {'backbone': self.backbone_wanted,
                                  'rpn_head': self.rpn_head_wanted,
                                  'bbox_head': self.bbox_head_wanted,
                                  'bbox_post_process': self.bbox_post_process_wanted}
        self.fasterRCNN_supervised = {'backbone': self.backbone_supervised, 
                                      'rpn_head': self.rpn_head_supervised,
                                      'bbox_head': self.bbox_head_supervised,
                                      'bbox_post_process': self.bbox_post_process_supervised}
        self._prepare_weights(weights_wanted_url, weights_supervised_url)
        

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
        bbox_post_process_wanted = create(cfg['bbox_post_process_wanted'], **kwargs)
        bbox_post_process_supervised = create(cfg['bbox_post_process_supervised'], **kwargs)
        return {
            'backbone_wanted': backbone_wanted,
            'backbone_supervised': backbone_supervised,
            'rpn_head_wanted': rpn_head_wanted,
            'rpn_head_supervised': rpn_head_supervised,
            'bbox_head_wanted': bbox_head_wanted,
            'bbox_head_supervised': bbox_head_supervised,
            'bbox_post_process_wanted': bbox_post_process_wanted,
            'bbox_post_process_supervised': bbox_post_process_supervised
        }

    def _predict(self, fasterRCNN):
        for component in ['backbone', 'rpn_head', 'bbox_head', 'bbox_post_process']:
            if component not in fasterRCNN.keys():
                raise ValueError(f'{component} not found in fasterRCNN.')
        # img_before = self.inputs['image'].clone()
        self.inputs = self.prompter(self.inputs)
        # img_after = self.inputs['image']
        # log_value(type(img_after), type(img_before))
        # log_value(img_after.shape, img_before.shape)
        # assert not paddle.equal(img_before[0][0][0][0], img_after[0][0][0][0])
        body_feats = fasterRCNN['backbone'](self.inputs)
        rois, rois_num, _ = fasterRCNN['rpn_head'](body_feats, self.inputs)
        preds, _ = fasterRCNN['bbox_head'](body_feats, rois, rois_num, None)
        im_shape = self.inputs['im_shape']
        scale_factor = self.inputs['scale_factor']
        bbox, bbox_num, nms_keep_idx = fasterRCNN['bbox_post_process'](preds, (rois, rois_num), im_shape, scale_factor)
        # rescale the prediction back to origin image
        bboxes, bbox_pred, bbox_num = fasterRCNN['bbox_post_process'].get_pred(bbox, bbox_num, im_shape, scale_factor)
        # log_value(bbox_feat)
        # log_value(type(bbox_feat), bbox_feat.shape)
        return bbox_pred, bbox_num, body_feats

    def _forward(self):
        # image = self.inputs['image']
        # log_value(type(image))
        # log_value(image.shape)
        if self.neck is not None:
            body_feats = self.neck(body_feats)
        if self.training:
            with paddle.no_grad():
                for _, value in self.fasterRCNN_supervised.items():
                    if isinstance(value, nn.Layer):
                        value.eval()
                bbox_pred, bbox_num, body_feats_supervised = self._predict(self.fasterRCNN_supervised)
                choose = bbox_pred[:, 1]
                gt_class = bbox_pred[:, :1]
                gt_bbox = bbox_pred[:, 2:]
                # threshold
                gt_class = gt_class[choose > 0.5]
                gt_bbox = gt_bbox[choose > 0.5]
            self.inputs['gt_class'] = (gt_class,)
            self.inputs['gt_bbox'] = (gt_bbox,)
            body_feats_wanted = self.backbone_wanted(self.inputs)
            rois, rois_num, rpn_loss = self.rpn_head_wanted(body_feats_wanted, self.inputs)  # USE GT
            bbox_loss, _ = self.bbox_head_wanted(body_feats_wanted, rois, rois_num,
                                          self.inputs)  # USE GT
            # log_value(len(body_feats_wanted), len(body_feats_supervised))
            # log_value(body_feats_wanted[0].shape, body_feats_supervised[0].shape)
            feat_loss = F.l1_loss(body_feats_wanted[0], body_feats_supervised[0])
            return rpn_loss, bbox_loss, feat_loss
            
        else:
            bbox_pred, bbox_num, _ = self._predict(self.fasterRCNN_wanted)
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
