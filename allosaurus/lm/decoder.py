from allosaurus.lm.inventory import *
from pathlib import Path
from itertools import groupby, product
import numpy as np


class PhoneDecoder:
    def __init__(self, model_path, inference_config):
        """
        This class is an util for decode both phones and words

        :param model_path:
        """

        # lm model path
        self.model_path = Path(model_path)

        self.config = inference_config

        # create inventory
        self.inventory = Inventory(model_path, inference_config)

        self.unit = self.inventory.unit

    def compute(
        self,
        logits,
        lang_id=None,
        topk=1,
        emit=1.0,
        timestamp=False,
        cutoff=0.0,
        topapprox=1.0,
        getproduct=False,
        hideblank=True,
        interleave=1
    ):
        """
        decode phones from logits

        :param logits: numpy array of logits
        :param emit: blank factor
        :return:
        """

        blank = "" if hideblank else "."

        # Adjust effective window shift and size
        eff_window_shift = self.config.window_shift / interleave
        eff_window_size = self.config.window_size # Window size remains the same (0.025s)

        # apply mask if lang_id specified
        mask = self.inventory.get_mask(lang_id, approximation=self.config.approximate)

        logits = mask.mask_logits(logits)

        decoded_seq = []
        prod_list = []
        for idx in range(len(logits)):
            logit = logits[idx]
            exp_prob = np.exp(logit - np.max(logit))
            probs = exp_prob / exp_prob.sum()

            top_phones = logit.argsort()[-topk:][::-1]
            top_probs = sorted(probs)[-topk:][::-1]
            best_prob = top_probs[0]

            stamp = (
                f"{eff_window_shift * idx:.3f} {eff_window_size:.3f} "
            )

            if topk == 1:
                phones_str = "".join(
                    token if idx != 0 else blank
                    for idx, token, prob in zip(
                        top_phones, mask.get_units(top_phones), top_probs
                    )
                    if prob >= max(cutoff, best_prob * topapprox)
                )
                if timestamp:
                    phones_str = stamp + phones_str

                if phones_str:
                    decoded_seq.append(phones_str)
            else:
                phone_prob_lst = [
                    f"{token} ({prob:.3f})" if idx != 0 else f"{blank} ({prob:.3f})"
                    for idx, token, prob in zip(
                        top_phones, mask.get_units(top_phones), top_probs
                    )
                    if prob >= max(cutoff, best_prob * topapprox)
                ]
                phones_str = " ".join(phone_prob_lst)

                if timestamp:
                    phones_str = stamp + phones_str

                if phones_str:
                    decoded_seq.append(phones_str)

                if getproduct:
                    prod_list.append(
                        [
                            str(token) if idx != 0 else blank
                            for idx, token, prob in zip(
                                top_phones, mask.get_units(top_phones), top_probs
                            )
                            if prob >= max(cutoff, best_prob * topapprox)
                        ]
                    )

        if timestamp:
            # Group identical consecutive phones and take the first timestamp
            collapsed_output = []
            current_phone = None
            start_stamp = None
            for line in decoded_seq:
                parts = line.split(' ')
                stamp = f"{float(parts[0]):.3f} {float(parts[1]):.3f}"
                phone = parts[2] if len(parts) > 2 else ''

                if phone != '.' and phone != current_phone: # . is blank token, collapse if not blank and different from current
                    if current_phone is not None:
                        collapsed_output.append(f"{start_stamp} {current_phone}")
                    current_phone = phone
                    start_stamp = stamp
                elif phone == '.' and current_phone is not None: # if current is a phone and current is blank, reset
                    collapsed_output.append(f"{start_stamp} {current_phone}")
                    current_phone = None
                    start_stamp = None

            if current_phone is not None:
                collapsed_output.append(f"{start_stamp} {current_phone}")

            phones = "\n".join(collapsed_output)
        elif topk == 1:
            # Apply collapsing for non-timestamped output
            # 1. Filter out all blank tokens ('.')
            filtered_seq = [k for k in decoded_seq if k != "."]
            # 2. Apply groupby to collapse consecutive identical phonemes
            collapsed_phones = [k for k, g in groupby(filtered_seq)]
            phones = "".join(collapsed_phones)
        elif getproduct:
            unique_collapsed_products = set()
            for poss_str_tuple in product(*prod_list):
                # Apply collapsing to each product path
                filtered_seq = [k for k in poss_str_tuple if k != "."]
                collapsed_path = "".join([k for k, g in groupby(filtered_seq)])
                unique_collapsed_products.add(collapsed_path)
            phones = "\n".join(sorted(list(unique_collapsed_products)))
        else:
            phones = "\n".join(decoded_seq)

        return phones
