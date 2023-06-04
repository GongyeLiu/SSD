import torch
import math
from guided_diffusion.unet import TimestepBlock
import abc


class FeatureStore(object):
    def __init__(self, keys):
        self.cur_fea_layer = 0

        self.store_keys = keys
        self.num_fea_layers = len(keys)

        self.step_store = self.get_empty_store()
        self.feature_store = []

        self.state = 'nothing'

    def set_state(self, state):
        assert state in ['store', 'inject', 'nothing']
        self.state = state

    def forward(self, feature, place_in_unet, layer_idx, is_attn=True):
        key = f"{place_in_unet}_{layer_idx}_attn" if is_attn else f"{place_in_unet}_{layer_idx}_fea"

        if self.state == 'nothing':
            return feature
        elif self.state == 'store':
            if key in self.store_keys:
                self.step_store[key] = feature
                self.cur_fea_layer += 1
            return feature
        elif self.state == 'inject':
            if key in self.store_keys:
                feature = self.get_feature(key)
            return feature

    def __call__(self, feature, place_in_unet, layer_idx, is_attn=True):
        feature = self.forward(feature, place_in_unet, layer_idx, is_attn)
        if self.cur_fea_layer == self.num_fea_layers:
            self.cur_fea_layer = 0
            self.feature_store.append(self.step_store)
            self.step_store = self.get_empty_store()
        return feature

    def get_feature(self, key):
        if self.cur_fea_layer == 0:
            self.step_store = self.feature_store.pop()
        feature = self.step_store[key]

        self.cur_fea_layer += 1
        if self.cur_fea_layer == self.num_fea_layers:
            self.cur_fea_layer = 0

        return feature

    def reset(self):
        self.cur_fea_layer = 0

        self.step_store = self.get_empty_store()
        self.feature_store = []
        self.state = 'nothing'

    def get_empty_store(self):
        return {key: [] for key in self.store_keys}


def register_feature_store(model, storer):
    def attn_forward(self, place_in_unet, layer_idx):
        def forward(qkv):
            """
            Apply QKV attention.

            :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
            :return: an [N x (H * C) x T] tensor after attention.
            """
            bs, width, length = qkv.shape
            assert width % (3 * self.n_heads) == 0
            ch = width // (3 * self.n_heads)
            q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
            scale = 1 / math.sqrt(math.sqrt(ch))
            weight = torch.einsum(
                "bct,bcs->bts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

            weight = storer(weight, place_in_unet, layer_idx, is_attn=True)

            a = torch.einsum("bts,bcs->bct", weight, v)
            return a.reshape(bs, -1, length)

        return forward

    def fea_forward(self, place_in_unet, layer_idx):
        def forward(x, emb):
            for layer in self:
                if isinstance(layer, TimestepBlock):
                    x = layer(x, emb)
                else:
                    x = layer(x)
            x = storer(x, place_in_unet, layer_idx, is_attn=False)
            return x

        return forward

    def register_recr(net_, place_in_unet, layer_idx):
        if net_.__class__.__name__ == 'QKVAttentionLegacy':
            net_.forward = attn_forward(net_, place_in_unet, layer_idx)

        if net_.__class__.__name__ == 'TimestepEmbedSequential':
            layer_idx += 1
            net_.forward = fea_forward(net_, place_in_unet, layer_idx)

        if hasattr(net_, 'children'):
            for net__ in net_.children():
                layer_idx = register_recr(net__, place_in_unet, layer_idx)

        return layer_idx

    sub_nets = model.named_children()
    for net in sub_nets:
        if "input" in net[0]:
            register_recr(net[1], "input", 0)
        elif "output" in net[0]:
            register_recr(net[1], "output", 0)
        elif "middle" in net[0]:
            register_recr(net[1], "middle", 0)


def register_feature_store_celeba(model, storer):
    def attn_forward(self, place_in_unet, layer_idx):
        # print(place_in_unet, layer_idx)
        def forward(x):
            h_ = x
            h_ = self.norm(h_)
            q = self.q(h_)
            k = self.k(h_)
            v = self.v(h_)

            # compute attention
            b, c, h, w = q.shape
            q = q.reshape(b, c, h * w)
            q = q.permute(0, 2, 1)  # b,hw,c
            k = k.reshape(b, c, h * w)  # b,c,hw
            w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
            w_ = w_ * (int(c) ** (-0.5))
            w_ = torch.nn.functional.softmax(w_, dim=2)

            # attend to values
            v = v.reshape(b, c, h * w)
            w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
            # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

            w_ = storer(w_, place_in_unet, layer_idx, is_attn=True)

            h_ = torch.bmm(v, w_)
            h_ = h_.reshape(b, c, h, w)

            h_ = self.proj_out(h_)

            return x + h_

        return forward

    def register_recr(net_, place_in_unet, layer_idx):
        # print(net_.__class__.__name__)
        if net_.__class__.__name__ == 'AttnBlock':
            net_.forward = attn_forward(net_, place_in_unet, layer_idx)
            layer_idx += 1

        # if net_.__class__.__name__ == 'Module':
            # net_.forward = fea_forward(net_, place_in_unet, layer_idx)

        if hasattr(net_, 'children'):
            for net__ in net_.children():
                layer_idx = register_recr(net__, place_in_unet, layer_idx)

        return layer_idx

    sub_nets = model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            register_recr(net[1], "input", 0)
        elif "up" in net[0]:
            register_recr(net[1], "output", 0)
        elif "mid" in net[0]:
            register_recr(net[1], "middle", 0)