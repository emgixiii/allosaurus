from allosaurus.am.utils import *
from pathlib import Path
from allosaurus.audio import read_audio, Audio
from allosaurus.pm.factory import read_pm
from allosaurus.am.factory import read_am
from allosaurus.lm.factory import read_lm
from allosaurus.bin.download_model import download_model
from allosaurus.model import resolve_model_name, get_all_models
from argparse import Namespace
from io import BytesIO


def read_recognizer(inference_config_or_name="latest", alt_model_path=None):
    if alt_model_path:
        if not alt_model_path.exists():
            download_model(inference_config_or_name, alt_model_path)
    # download specified model automatically if no model exists
    if len(get_all_models()) == 0:
        download_model("latest", alt_model_path)

    # create default config if input is the model's name
    if isinstance(inference_config_or_name, str):
        model_name = resolve_model_name(inference_config_or_name, alt_model_path)
        inference_config = Namespace(
            model=model_name, device_id=-1, lang="ipa", approximate=False, prior=None, interleave=1
        )
    else:
        assert isinstance(inference_config_or_name, Namespace)
        inference_config = inference_config_or_name

    if alt_model_path:
        model_path = alt_model_path / inference_config.model
    else:
        model_path = Path(__file__).parent / "pretrained" / inference_config.model

    if inference_config.model == "latest" and not model_path.exists():
        download_model(inference_config, alt_model_path)

    assert model_path.exists(), f"{inference_config.model} is not a valid model"

    # create pm (pm stands for preprocess model: audio -> feature etc..)
    pm = read_pm(model_path, inference_config)

    # create am (acoustic model: feature -> logits )
    am = read_am(model_path, inference_config)

    # create lm (language model: logits -> phone)
    lm = read_lm(model_path, inference_config)

    return Recognizer(pm, am, lm, inference_config)


class Recognizer:
    def __init__(self, pm, am, lm, config):

        self.pm = pm
        self.am = am
        self.lm = lm
        self.config = config

    def is_available(self, lang_id):
        # check whether this lang id is available

        return self.lm.inventory.is_available(lang_id)

    def recognize(
        self,
        filename,
        lang_id="ipa",
        topk=1,
        emit=1.0,
        timestamp=False,
        cutoff=0.0,
        topapprox=0.0,
        getproduct=False,
        hideblank=False,
    ):
        # recognize a single file

        # filename check (skipping for BytesIO objects)
        if not isinstance(filename, BytesIO):
            assert str(filename).endswith(".wav"), (
                "only wave file is supported in allosaurus"
            )

        # load wav audio
        audio = read_audio(filename)
        
        # get interleave factor from config
        n = getattr(self.config, 'interleave', 1)
        if n < 1: n = 1

        # original window shift (in seconds)
        window_shift = self.lm.config.window_shift
        
        # calculate sample offsets
        offset_samples = [int(audio.sample_rate * (window_shift / n) * i) for i in range(n)]
        
        all_feats = []
        feat_lengths = []
        
        for offset in offset_samples:
            # create a temporary audio object with offset samples
            if offset > 0:
                offset_audio = Audio(audio.samples[offset:], audio.sample_rate)
            else:
                offset_audio = audio
            
            # extract feature for this offset
            feat = self.pm.compute(offset_audio)
            all_feats.append(feat)
            feat_lengths.append(feat.shape[0])

        # find max length for padding if needed (though they should be very similar)
        max_len = max(feat_lengths)
        dim = all_feats[0].shape[1]
        
        # create batch tensor
        batch_feats = np.zeros((n, max_len, dim), dtype=np.float32)
        for i, feat in enumerate(all_feats):
            batch_feats[i, :feat_lengths[i], :] = feat
        
        batch_feat_len = np.array(feat_lengths, dtype=np.int32)

        tensor_batch_feat, tensor_batch_feat_len = move_to_tensor(
            [batch_feats, batch_feat_len], self.config.device_id
        )

        tensor_batch_lprobs = self.am(tensor_batch_feat, tensor_batch_feat_len)

        if self.config.device_id >= 0:
            batch_lprobs = tensor_batch_lprobs.cpu().detach().numpy()
        else:
            batch_lprobs = tensor_batch_lprobs.detach().numpy()

        # batch_lprobs shape: (n, T, phones)
        # Interleave the results
        # We take indices [0,0], [1,0], [2,0], [0,1], [1,1], [2,1]...
        # We need to be careful about different T if offsets cause different frame counts
        min_t = min(feat_lengths)
        
        # Interleave logits
        # Reshape to (T, n, phones) then (T*n, phones)
        interleaved_lprobs = batch_lprobs[:, :min_t, :].transpose(1, 0, 2).reshape(-1, batch_lprobs.shape[2])

        token = self.lm.compute(
            interleaved_lprobs,
            lang_id,
            topk,
            emit=emit,
            timestamp=timestamp,
            cutoff=cutoff,
            topapprox=topapprox,
            getproduct=getproduct,
            hideblank=hideblank,
            interleave=n
        )
        return token
