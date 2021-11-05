from __future__ import division

# Import from third library
from easydict import EasyDict

from eod.utils.general.registry_factory import DATALOADER_REGISTRY
from eod.data.data_loader import BaseDataLoader


__all__ = ['MOTDataLoader']


@DATALOADER_REGISTRY.register('mot')
class MOTDataLoader(BaseDataLoader):

    def _collate_fn(self, batch):
        """
        Form a mini-batch from list of data of :meth:`~root.datasets.base_dataset.BaseDataset.__getitem__`

        Arguments:
            - batch (:obj:`list` of data): type of data depends on output of Dataset.
              For :class:`~root.datasets.coco_dataset.CocoDataset`,
              :class:`~root.datasets.voc_dataset.VocDataset` and :class:`~root.datasets.custom_dataset.CustomDataset`,
              data is a Dictionary.

        Returns:
            - output (:obj:`dict`)

        Output example::

            {
                # (FloatTensor): [B, 3, max_h, max_w], RGB format
                'image': ..,
                # (list of FloatTensor): [B, 5], (resized_h, resize_w, scale_factor, origin_h, origin_w)
                'image_info': ..,
                # (list of FloatTensor): [B] [num_gts, 5] or None
                'gt_bboxes': ..,
                # (list of FloatTensor): [B] [num_igs, 4] or None
                'ig_bboxes': ..,
                #(list of FloatTensor): [B] [num_gts, k, 3] or None
                'gt_keyps': ..,
                #(list of list of list of ndarray): [B] [num_gts] [polygons] or None
                'gt_masks': ..,
                # (FloatTensor) [B, max_num_gts, num_grid_points(9), 3] or None
                'gt_grids': ..,
                # (FloatTensor) [B, 1, max_h, max_w], semantic mask
                'gt_semantic_seg': ..,
                # filename of images
                'filenames': ..,
                # label of negative sample (default = 0)
                'neg_targets': ..
            }
        """
        main = self._process([one['main'] for one in batch])
        ref = [self._process([one['ref'][i] for one in batch]) for i in range(self.dataset.ref_num)]
        return {'main': main, 'ref': ref}

    def _process(self, batch):
        images = [_['image'] for _ in batch]
        image_info = [_['image_info'] for _ in batch]
        filenames = [_['filename'] for _ in batch]
        image_ids = [_['image_id'] for _ in batch]
        flipped = [_['flipped'] for _ in batch]
        neg_targets = [_.get('neg_target', 0) for _ in batch]
        image_sources = [_.get('image_source', 0) for _ in batch]

        gt_bboxes = [_.get('gt_bboxes', None) for _ in batch]
        gt_ignores = [_.get('gt_ignores', None) for _ in batch]

        output = EasyDict({
            'image': images,
            'image_info': image_info,
            'image_id': image_ids,
            'filenames': filenames,
            'flipped': flipped,
            'neg_targets': neg_targets,
            'image_sources': image_sources,
        })

        output['gt_bboxes'] = gt_bboxes if gt_bboxes[0] is not None else None
        output['gt_ignores'] = gt_ignores if gt_ignores[0] is not None else None
        output = self.pad(output)
        return output
