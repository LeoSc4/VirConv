from .detector3d_template import Detector3DTemplate
import time
class VoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:

            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # print('DEBUG - Inference -> Starting Post_processing for infered results ')

            pred_dicts, recall_dicts, = self.post_processing(batch_dict) #post processing for inference
            return pred_dicts, recall_dicts, batch_dict


    # Highest level function for loss calculation
    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss() #get the RPN loss from the DenseHead which is generating proposals (anchors)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict) #get the RCNN loss for the multiple cascades in VirConv-T

        loss =  loss_rpn + loss_rcnn # equal weighted sum of the two losses
        return loss, tb_dict, disp_dict

