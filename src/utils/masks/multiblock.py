import math

from multiprocessing import Value

from logging import getLogger

import torch

logger = getLogger()

class MaskCollator(object):

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        num_enc_masks=1,
        num_pred_masks=2,
        min_keep=4,
        **kwargs
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.min_keep = min_keep  # minimum number of patches to keep
        self.aspect_ratio = aspect_ratio
        self.nenc = num_enc_masks
        self.npred = num_pred_masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # --
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # --
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # --
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size
        valid_mask = False

        def constrain_mask(mask, tries=0):
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        tries = 0
        timeout = og_timeout = 20
        while not valid_mask:
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        enc_is_bigger = e_size[0]*e_size[1] > p_size[0]*p_size[1]

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):
            masks_p, masks_C = [], []
            masks_e = []

            if enc_is_bigger:
                for _ in range(self.npred):
                    mask, mask_C = self._sample_block_mask(p_size)
                    masks_p.append(mask)
                    masks_C.append(mask_C)
                    min_keep_pred = min(min_keep_pred, len(mask))
                collated_masks_pred.append(masks_p)

                for _ in range(self.nenc):
                    mask, _ = self._sample_block_mask(e_size, acceptable_regions=masks_C)
                    masks_e.append(mask)
                    min_keep_enc = min(min_keep_enc, len(mask))
                collated_masks_enc.append(masks_e)

            else:
                for _ in range(self.nenc):
                    mask, mask_C = self._sample_block_mask(e_size)
                    masks_e.append(mask)
                    masks_C.append(mask_C)
                    min_keep_enc = min(min_keep_enc, len(mask))
                collated_masks_enc.append(masks_e)

                for _ in range(self.npred):
                    mask, _ = self._sample_block_mask(p_size, acceptable_regions=mask_C)
                    masks_p.append(mask)
                    masks_C.append(mask_C)
                    min_keep_pred = min(min_keep_pred, len(mask))
                collated_masks_pred.append(masks_p)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred

    def get_info(self):
        info = {
            'input_size': (self.height * self.patch_size, self.width * self.patch_size),
            'patch_size': self.patch_size,
            'enc_mask_scale': self.enc_mask_scale,
            'pred_mask_scale': self.pred_mask_scale,
            'aspect_ratio': self.aspect_ratio,
            'nenc': self.nenc,
            'npred': self.npred,
            'min_keep': self.min_keep,
        }
        return info