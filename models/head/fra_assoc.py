import torch
import torch.nn as nn
import torch.nn.functional as F


class AssocModel(nn.Module):

    def __init__(self, no_fr=False, framerate_mode='spec', ignore_appr=False):
        super().__init__()
        self.sin_div = 256
        self.no_fr = no_fr
        self.ignore_appr = ignore_appr
        self.aff_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        if not self.no_fr:
            assert framerate_mode in [
                'spec', 'auto', 'complex'], f'Invalid framerate mode {framerate_mode}, must be one of spec, auto, complex'
            self.framerate_mode = framerate_mode
            if self.framerate_mode == 'spec':
                self.att_net = nn.Sequential(
                    nn.Linear(self.sin_div, 128),
                    nn.ReLU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 64),
                )
            if self.framerate_mode == 'auto':
                self.att_net = nn.Sequential(
                    nn.Linear(300, 128),
                    nn.ReLU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 64),
                )
            if self.framerate_mode == 'complex':
                self.att_net = nn.Sequential(
                    nn.Linear(self.sin_div + 300, 128),
                    nn.ReLU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 64),
                )

    def sin_embedding(self, frame_rate, device='cuda', dtype=torch.float32):
        framerates_emb = torch.arange(self.sin_div, device=device, dtype=dtype).reshape(
            1, self.sin_div).repeat(frame_rate.shape[0], 1)
        framerates_emb = torch.cos(
            framerates_emb * frame_rate.unsqueeze(1) * 1.5 / self.sin_div)
        return framerates_emb

    def forward(self, data):
        if self.ignore_appr:
            aff_feats = torch.cat([data['features'][:, 1:2], data['features'].new_zeros(
                data['features'].shape[0], 1), data['features'][:, 3:]], dim=1)
        else:
            aff_feats = data['features'][:, 1:]
        framerates = data['features'][:, 0]

        aff_feats = self.aff_net(aff_feats)
        if not self.no_fr:
            if self.framerate_mode == 'spec':
                frame_rate_embeddings = self.sin_embedding(
                    framerates, device=aff_feats.device, dtype=aff_feats.dtype)
            if self.framerate_mode == 'auto':
                frame_rate_embeddings = data['ctrl']
            if self.framerate_mode == 'complex':
                data_ctrl = data['ctrl']
                if data_ctrl.shape[0] != aff_feats.shape[0]:
                    data_ctrl = data['ctrl'].repeat(aff_feats.shape[0], 1)
                frame_rate_embeddings = torch.cat(
                    [self.sin_embedding(framerates, device=aff_feats.device, dtype=aff_feats.dtype),
                     data_ctrl], dim=1)
            aff_att = self.att_net(frame_rate_embeddings)
            data['attention'] = aff_att
            affinities = (aff_feats * aff_att).sum(dim=1)
        else:
            affinities = (aff_feats).sum(dim=1)
        data['pred'] = affinities
        return data


class AssocModelMTB(nn.Module):

    def __init__(self, frs=[1, 2, 4, 8, 16, 25, 36, 50, 75], ignore_appr=False):
        super().__init__()
        self.ignore_appr = ignore_appr
        self.frs = list(frs)
        self.aff_nets = nn.ModuleList()
        for fr in frs:
            sub_ = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            )
            self.aff_nets.append(sub_)

    def find_index(self, fr):
        mi = abs(self.frs[0] - fr)
        mind = 0
        for i in range(len(self.frs)):
            if abs(self.frs[i] - fr) < mi:
                mi = abs(self.frs[i] - fr)
                mind = i
        return mind

    def forward(self, data):
        if self.ignore_appr:
            aff_feats = torch.cat([data['features'][:, 1:2], data['features'].new_zeros(
                data['features'].shape[0], 1), data['features'][:, 3:]], dim=1)
        else:
            aff_feats = data['features'][:, 1:]
        framerates = data['features'][:, 0]
        preds = framerates.new_zeros(framerates.shape[0],)
        for fr in framerates.unique():
            fmask = framerates == fr
            k = self.find_index(fr)
            # print(fr, fmask.sum(), k)
            aff_feats_ = aff_feats[fmask]
            aff_feats_ = self.aff_nets[k](aff_feats_)
            preds[fmask] = (aff_feats_).sum(dim=1)

        data['pred'] = preds.contiguous()
        return data


# if __name__ == '__main__':
#     model = AssocModelMTB()
#     a = torch.randint(1, 100, (100, 1))
#     b = torch.randn((100, 4))
#     c = torch.cat([a.float(), b.float()], dim=1).float()
#     d = model({'features': c})
#     print(d)


class mNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(mNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, data):
        f = data['features_0']
        f = F.relu(self.fc1(f))
        pred = self.fc2(f)
        data['pred'] = pred
        return data


class mNN4(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)

    def forward(self, data):
        f = data['features_0']
        f = self.bn1(F.relu(self.fc1(f)))
        f = self.bn2(F.relu(self.fc2(f)))
        f = self.bn3(F.relu(self.fc3(f)))
        pred = self.fc4(f)
        data['pred'] = pred
        return data


class ClsRegNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.fc1_cls = nn.Linear(input_size, hidden_size)
        self.fc2_cls = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3_cls = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4_cls = nn.Linear(hidden_size // 4, 2)
        self.bn1_cls = nn.BatchNorm1d(hidden_size)
        self.bn2_cls = nn.BatchNorm1d(hidden_size // 2)
        self.bn3_cls = nn.BatchNorm1d(hidden_size // 4)

    def forward(self, data):
        f = data['features']
        f = self.bn1(F.relu(self.fc1(f)))
        f = self.bn2(F.relu(self.fc2(f)))
        f = self.bn3(F.relu(self.fc3(f)))
        pred = self.fc4(f)
        data['pred_reg'] = pred
        f = data['features_0']
        f = self.bn1_cls(F.relu(self.fc1_cls(f)))
        f = self.bn2_cls(F.relu(self.fc2_cls(f)))
        f = self.bn3_cls(F.relu(self.fc3_cls(f)))
        pred_cls = torch.sigmoid(self.fc4_cls(f))
        data['pred_cls'] = pred_cls
        return data


class MarkovNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)

        self.fc = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.trans_weight = nn.Parameter(torch.FloatTensor(
            2, 2, hidden_size))
        self.trans_bias = nn.Parameter(torch.FloatTensor(
            2, 2))
        nn.init.normal_(self.trans_weight.data)
        nn.init.zeros_(self.trans_bias.data)

    def forward(self, data):
        f = data['features']
        f = self.bn1(F.relu(self.fc1(f)))
        f = self.bn2(F.relu(self.fc2(f)))
        f = self.bn3(F.relu(self.fc3(f)))
        pred_reg = self.fc4(f)

        f = data['features_0']
        f = F.relu(self.fc(f))
        f = self.bn(f)
        s = data['states']
        k, k, d = self.trans_weight.shape
        treshaped = self.trans_weight.reshape(k*k, d).T
        trans_f = (f.matmul(treshaped).reshape(-1, k, k) +
                   self.trans_bias.reshape(1, k, k)).softmax(dim=1)
        pred_cls = torch.bmm(trans_f, s.unsqueeze(-1)).squeeze(-1)

        data['pred_cls'] = pred_cls
        data['pred_reg'] = pred_reg
        return data


class mRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(mRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, data):
        f = data['features']
        h0 = torch.zeros(1, f.size(0), self.hidden_size)
        out, hn = self.rnn(f, h0)
        pred = self.fc(out[:, -1, :])
        data['pred'] = pred
        return data


class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(mLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, data):
        f = data['features']
        h0 = torch.zeros(1, f.size(0), self.hidden_size)
        c0 = torch.zeros(1, f.size(0), self.hidden_size)
        out, (hn, cn) = self.lstm(f, (h0, c0))
        pred = self.fc(out[:, -1, :])
        data['pred'] = pred
        return data


class Markov(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Markov, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.trans_weight = nn.Parameter(torch.FloatTensor(
            output_size, output_size, hidden_size))
        self.trans_bias = nn.Parameter(torch.FloatTensor(
            output_size, output_size))
        nn.init.normal_(self.trans_weight.data)
        nn.init.zeros_(self.trans_bias.data)
        # self.fc2

    def forward(self, data):
        #  P_{tran} = activation(M * F) F(D, 1) M(K, K, D) (K, K)
        #  Pi_t (K,) = P_{tran} * Pi_{t-1} (K,)

        f = data['features']
        f = F.relu(self.fc(f))
        f = self.bn(f)
        s = data['states']
        k, k, d = self.trans_weight.shape
        treshaped = self.trans_weight.reshape(k*k, d).T
        trans_f = (f.matmul(treshaped).reshape(-1, k, k) +
                   self.trans_bias.reshape(1, k, k)).softmax(dim=1)
        ns = torch.bmm(trans_f, s.unsqueeze(-1)).squeeze(-1)
        data['pred'] = ns
        return data

    def forward2(self, data):
        n = data['features_n']
        assert n > 1
        f = data['features'][0]
        out = []
        # M = (D, D*4) (D*4, ) => (D, )
        # (D, D) (D, ) => (D, )    .....  *4,    (D, )*4 => (D, )
        # pred = [self.m1 * f_0 + self.m2 * f_1 + self.m3 * f_2, self.m4 * f_total])  eq(1)
        # pred = self.M * [f_0, f_1, f_2, f_total] eq(2)
        # pred = self.M * F
        for i in range(1, n):
            o = self.func(f, data['features'][i])
            out.append(o)
            f = o
        data['pred'] = out
        return data


class mTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(mTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, data):
        f = data['features']
        f = self.transformer(f)
        pred = self.fc(f)
        data['pred'] = pred
        return data

# v_t(x(t)) = g(t, x(t))

# loss = f(y0(N, 3), a).mean() =

# v_t(x(t)) = min_a{ f(v_{t+1}(x(t+1)), a) }

# v_t(x(t)) = x(t)^a

# v_{t-1}(x;a) = NN_1(x(t-1);a)


# v_t(x) = y = a(x) x


# a = NN_2(x;f0)
