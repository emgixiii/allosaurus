from allosaurus.lm.inventory import *
from pathlib import Path
from itertools import groupby, product
import numpy as np


def apply_majority_filter(seq, n):
    if n <= 1:
        return seq
    res = []
    length = len(seq)
    for i in range(length):
        start = max(0, i - n // 2)
        end = min(length, start + n)
        if end == length:
            start = max(0, end - n)
        window = seq[start:end]
        counts = {}
        for item in window:
            counts[item] = counts.get(item, 0) + 1
        max_item = seq[i]
        max_count = 0
        for item, count in counts.items():
            if count > max_count:
                max_count = count
                max_item = item
            elif count == max_count and item == seq[i]:
                max_item = item
        res.append(max_item)
    return res


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
        interleave=1,
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
        eff_window_size = (
            self.config.window_size
        )  # Window size remains the same (0.025s)

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

            stamp = f"{eff_window_shift * idx:.3f} {eff_window_size:.3f} "

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
            if interleave > 1:
                # Extract phones, smooth them, and update decoded_seq
                phones = [line.split(" ", 2)[2] for line in decoded_seq]
                smoothed_phones = apply_majority_filter(phones, interleave)
                new_decoded_seq = []
                for line, smoothed_phone in zip(decoded_seq, smoothed_phones):
                    parts = line.split(" ", 2)
                    new_decoded_seq.append(f"{parts[0]} {parts[1]} {smoothed_phone}")
                decoded_seq = new_decoded_seq

            # Group identical consecutive phones and calculate start and end timestamps
            collapsed_output = []
            current_phone = None
            start_time = None
            end_time = None

            for line in decoded_seq:
                parts = line.split(" ", 2)
                if len(parts) < 3:
                    continue

                f_start = float(parts[0])
                f_dur = float(parts[1])
                f_end = f_start + f_dur
                phone = parts[2]

                if phone == "." or phone == "":
                    if current_phone is not None:
                        collapsed_output.append(f"{start_time:.3f} {end_time:.3f} {current_phone}")
                        current_phone = None
                    continue

                if phone != current_phone:
                    if current_phone is not None:
                        collapsed_output.append(f"{start_time:.3f} {end_time:.3f} {current_phone}")
                    current_phone = phone
                    start_time = f_start
                    end_time = f_end
                else:
                    # same phone, update end_time
                    end_time = f_end

            if current_phone is not None:
                collapsed_output.append(f"{start_time:.3f} {end_time:.3f} {current_phone}")

            phones = "\n".join(collapsed_output)
        elif topk == 1:
            # Apply majority filter for robustness if interleave > 1
            if interleave > 1:
                decoded_seq = apply_majority_filter(decoded_seq, interleave)

            # Apply collapsing for non-timestamped output
            # 1. Filter out all blank tokens ('.')
            filtered_seq = [k for k in decoded_seq if k != "."]
            # 2. Apply groupby to collapse consecutive identical phonemes
            collapsed_phones = [k for k, g in groupby(filtered_seq)]
            phones = "".join(collapsed_phones)
        elif getproduct:
            unique_collapsed_products = set()
            for poss_str_tuple in product(*prod_list):
                # Apply majority filter for robustness if interleave > 1
                if interleave > 1:
                    poss_str_tuple = apply_majority_filter(poss_str_tuple, interleave)

                # Apply collapsing to each product path
                filtered_seq = [k for k in poss_str_tuple if k != "."]
                collapsed_path = "".join([k for k, g in groupby(filtered_seq)])
                unique_collapsed_products.add(collapsed_path)
            phones = "\n".join(sorted(list(unique_collapsed_products)))
        else:
            phones = "\n".join(decoded_seq)

        return phones
