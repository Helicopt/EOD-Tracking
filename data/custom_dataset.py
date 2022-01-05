import json
from eod.utils.general.log_helper import default_logger as logger
from eod.tasks.det.data.datasets.custom_dataset import CustomDataset

from eod.utils.general.registry_factory import DATASET_REGISTRY

from ..utils.read_helper import read_lines

@DATASET_REGISTRY.register('det_custom')
class DetCustomDataset(CustomDataset):
    def _normal_init(self):
        if not isinstance(self.meta_file, list):
            self.meta_file = [self.meta_file]
        for idx, meta_file in enumerate(self.meta_file):
            for data in read_lines(meta_file):
                if self.label_mapping is not None:
                    data = self.set_label_mapping(data, self.label_mapping[idx], 0)
                    data['image_source'] = idx
                self.metas.append(data)
                if 'image_height' not in data or 'image_width' not in data:
                    logger.warning('image size is not provided, '
                                    'set aspect grouping to 1.')
                    self.aspect_ratios.append(1.0)
                else:
                    self.aspect_ratios.append(data['image_height'] / data['image_width'])