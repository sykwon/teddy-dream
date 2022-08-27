import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from torchcontrib.optim import SWA
from tqdm import tqdm
from collections.abc import Iterable

import src.util as ut
from src.estimator import Estimator

is_debug = False


class RNNdataset(Dataset):
    def __init__(self, x, y, prfx=False):
        self.x = x
        self.y = y
        self.prfx = prfx
        if self.prfx:
            assert len(x) == len(y), \
                f"The numbers of data and labels should be the same, but {len(x)} and {len(y)} are given"
            self.y = []
            for y_elem in y:
                y_elem = list(zip(*y_elem))
                self.y.append(y_elem)
            self.max_d = len(y[0][0]) - 1
            self._len = len(y) * (self.max_d + 1)
        else:
            assert len(x[0]) == len(y), \
                f"The numbers of data and labels should be the same, but {len(x[0])} and {len(y)} are given"
            self._len = len(y)

    def __getitem__(self, item):
        if self.prfx:
            item, d = item // (self.max_d + 1), item % (self.max_d + 1)
            # assert len(self.x[item]) == len(self.y[item][d])  # debug hook
            return self.x[item], d, self.y[item][d]
        else:
            return self.x[0][item], self.x[1][item], self.y[item]

    def __len__(self):
        return self._len


def collate_fn_RNN_prfx(batch):
    x_seq, x_d, y = list(zip(*batch))
    n_batch = len(x_seq)
    x_len = [len(x) for x in x_seq]
    x_seq = ut.keras_pad_sequences(x_seq, padding='post')
    y_pad = ut.keras_pad_sequences(y, padding='post')
    x_seq, x_d, x_len, y = torch.LongTensor(x_seq), torch.LongTensor(x_d), torch.LongTensor(x_len), torch.FloatTensor(
        y_pad)

    return (x_seq, x_d, x_len), y


def collate_fn_RNN(batch):
    x_seq, x_d, y = list(zip(*batch))
    x_len = [len(x) for x in x_seq]
    x_seq = ut.keras_pad_sequences(x_seq, padding='post')
    x_seq, x_d, x_len, y = torch.LongTensor(x_seq), torch.LongTensor(x_d), torch.LongTensor(x_len), torch.FloatTensor(y)
    return (x_seq, x_d, x_len), y


class DREAMEstimator(Estimator):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf
        self.class_name = str(self.__class__).split("'")[1].split(".")[1]
        self.name = conf.name
        self.save_path = None
        self.patience = conf.patience
        self.best_epoch = 0
        self.last_epoch = 0
        self.sep_emb = conf.sep_emb
        self.multi = conf.multi
        self.prfx = conf.prfx
        self.btS = conf.btS
        self.btA = conf.btA
        self.l2 = conf.l2
        self.seq_out = self.prfx or self.btS or self.btA
        self.clip_gr = conf.clip_gr

    def build(self, train_data, valid_data=None, test_data=None, over_write=False):
        #
        # econf = self.conf
        # dconf = dfact.conf
        # qconf = dfact.query_strategy.conf
        # ntrain = econf.ntrain
        # nvalid = econf.nvalid
        # ntotal = ntrain + nvalid
        # ntest = econf.ntest
        # self.lr = lr = econf.lr
        # self.bs = bs = econf.bs
        #
        # gen_total = dfact.get_query(ntotal)
        # gen_test = dfact.get_test_query(ntest)
        # pos_queries = None
        # engram = None
        # n = dconf.n
        # N = 5
        # PT = 20
        # L = 20
        # dname = dconf.name
        # l_range = qconf.l_range
        # w = qconf.w
        # neg_ratio = econf.neg_ratio
        # seed = econf.seed
        # n_pos, n_neg = ut.split_total_with_ratio(ntotal, neg_ratio)
        #
        # is_count = True
        # is_sep = False
        # pos_queries = ut.load_filter_empty_query(pos_queries, engram, n=n, N=N, PT=PT, dname=dname, is_pos=True,
        #                                          seed=None,
        #                                          l_range=l_range, w=w)
        # neg_queries = None
        # if n_pos > len(pos_queries):
        #     n_pos = len(pos_queries)
        #     n_neg = round(n_pos * neg_ratio)
        #     ntotal = n_pos + n_neg
        #     ntrain = ntotal - nvalid
        # total_q, total_x, total_y = ut.load_total_train_data(pos_queries, neg_queries, db, engram, n=n, N=N, PT=PT, L=L,
        #                                                      dname=dname,
        #                                                      ntotal=ntotal, n_pos=n_pos, is_sep=is_sep,
        #                                                      is_count=is_count, neg_ratio=neg_ratio, seed=0)
        # del total_x
        #
        # data_total = []
        # for (q_s, d), y in zip(total_q, total_y):
        #     x = ut.string_query_encoding(q_s, d, char_dict)
        #     data_total.append((x, y))
        #
        # data_train, data_valid = data_total[:ntrain], data_total[ntrain:]

        dfact = self.db_factory
        db = dfact.get_db()
        self.char_dict = ut.char_dict_from_db(db, max_char=self.conf.max_char)
        self.n_char = n_char = len(self.char_dict)
        save_path = self.save_path
        max_len = max([len(x[0]) for x in train_data])
        assert save_path is not None, "save path should be given"

        # configuration
        conf = self.conf
        pred_hs = conf.h_dim
        cell_hs = conf.cs
        embed_size = conf.es
        seed = conf.seed
        num_rnn_layer = conf.layer
        num_pred_layer = conf.pred_layer
        n_channel = conf.n_channel
        max_d = conf.max_d
        delta = conf.delta
        if delta is not None:
            max_d = 0
        prfx = self.prfx
        btS = self.btS
        btA = self.btA

        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # assert torch.cuda.is_available()
        if device.type == "cpu":
            warnings.warn("check whether CUDA is available")

        # for reproducibility

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        self.model = RNN_module(n_char + 1, pred_hs, cell_hs, embed_size, num_pred_layer=num_pred_layer,
                                max_d=max_d,  num_rnn_layer=num_rnn_layer,
                                prfx=prfx, max_len=max_len, btS=btS, btA=btA)  # +1 means [UNK] token
        # simple_input = (torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]), torch.LongTensor([12]))

        ut.print_torch_summarize(self.model)
        if not over_write and os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path, map_location=torch.device(self.device)))
            if device.type == "cuda":
                self.model.to(device)
        else:
            # self.train(train_data, valid_data, test_data)
            self.train(train_data, valid_data)

    def model_size(self, *args):
        size = os.path.getsize(self.save_path)
        return size

    def _encode_data(self, raw_data):
        if self.seq_out:
            if len(raw_data[0][1]) > 1 and isinstance(raw_data[0][1][1], list):  # train mode
                qs_list, y_train = list(zip(*[(x[0], x[1]) for x in raw_data]))
                # encoding
                x_train = []
                for q_s in qs_list:
                    if self.multi or self.sep_emb:
                        x_train.append(ut.string_query_encoding(q_s, self.char_dict))
                    else:
                        raise NotImplementedError
                return x_train, y_train
            else:  # test mode
                raw_data_bak = raw_data
                raw_data = []
                for rs in raw_data_bak:
                    for d, c in enumerate(rs[1]):
                        raw_data.append([rs[0], d, c])

                qs_list, qd_list, y_train = list(zip(*raw_data))
                x_train = []
                for q_s, q_d in zip(qs_list, qd_list):
                    if self.multi or self.sep_emb:
                        x_train.append(ut.string_query_encoding(q_s, self.char_dict))
                    else:
                        x_train.append(ut.string_query_encoding(q_s, self.char_dict, d=q_d))
                return (x_train, qd_list), y_train
        else:
            if len(raw_data[0]) == 2:
                raw_data_bak = raw_data
                raw_data = []
                for rs in raw_data_bak:
                    for d, c in enumerate(rs[1]):
                        raw_data.append([rs[0], d, c])
            qs_list, qd_list, y_train = list(zip(*raw_data))
            x_train = []
            for q_s, q_d in zip(qs_list, qd_list):
                if self.multi or self.sep_emb:
                    x_train.append(ut.string_query_encoding(q_s, self.char_dict))
                else:
                    x_train.append(ut.string_query_encoding(q_s, self.char_dict, d=q_d))

            return (x_train, qd_list), y_train

    def _device_batch(self, batch):
        device = self.device
        # if self.conf.prfx:
        #     (x_seq, x_len), y_batch = batch  # unpack batch
        #
        #     # send data to device (CPU or GPU)
        #     x_seq = x_seq.to(device)
        #     # x_len = x_len.to(device)
        #     y_batch = y_batch.to(device)
        #
        #     x_batch = x_seq, x_len
        # else:
        (x_seq, x_d, x_len), y_batch = batch  # unpack batch

        # send data to device (CPU or GPU)
        x_seq = x_seq.to(device)
        x_len = x_len.to(device)
        x_d = x_d.to(device)
        y_batch = y_batch.to(device)

        x_batch = x_seq, x_d, x_len

        return x_batch, y_batch

    def dataloader(self, x_data, y_data, shuffle):
        bs = self.conf.bs
        assert bs >= 1

        prfx_train = (self.seq_out is True) and isinstance(y_data[0], Iterable)
        ds_data = RNNdataset(x_data, y_data, prfx=prfx_train)
        if prfx_train:
            dl_data = DataLoader(ds_data, batch_size=bs, shuffle=shuffle, collate_fn=collate_fn_RNN_prfx)
        else:
            dl_data = DataLoader(ds_data, batch_size=bs, shuffle=shuffle, collate_fn=collate_fn_RNN)
        return dl_data

    def q_error(self, y, y_hat, lengths=None):
        if (self.seq_out is True) and isinstance(y_hat[0], Iterable):
            if self.prfx:
                assert lengths is not None
                y_pack = ut.pack_padded_sequence(y, lengths)
                y_hat_pack = ut.pack_padded_sequence(y_hat, lengths)
            else:
                y_pack, y_hat_pack = ut.pack_padded_sequence_by_true_values(y, y_hat)
            q_err = ut.q_error(y_pack, y_hat_pack)
        else:
            q_err = ut.q_error(y, y_hat)
        return q_err

    def train(self, train_data, valid_data=None, test_data=None):
        # ---- test same train data ---- #
        # print("while test ")
        # import pickle as pkl
        # # pkl_path, test_data = "train.pkl", train_data
        # pkl_path, test_data = "valid.pkl", valid_data
        # if os.path.exists(pkl_path):
        #     test_data_back = pkl.load(open(pkl_path, "rb"))
        #     for i in range(len(test_data[0])):
        #         x_s, x_d = (test_data[0][i], test_data[1][i])
        #         b_s, b_d = (test_data_back[0][i], test_data_back[1][i])
        #         assert x_s == b_s, f"{x_s}, {b_s}"
        #         assert x_d == b_d
        # else:
        #     pkl.dump(test_data, open(pkl_path, "wb"))
        # exit()
        # ---- end test ---------- #

        model = self.model
        # model.to(self.device)
        model.cuda()
        model.train()
        econf = self.conf
        lr = econf.lr
        is_swa = econf.swa
        max_epoch = econf.max_epoch
        save_path = self.save_path
        logdir = self.logdir
        assert save_path is not None, "save_path should be assigned"
        assert logdir is not None, "log_path should be assigned"
        curr_best_path = os.path.dirname(save_path) + "/_curr_best.pt"

        x_train, y_train = self._encode_data(train_data)
        if valid_data:
            x_valid, y_valid = self._encode_data(valid_data)
        if test_data:
            x_test, y_test = self._encode_data(test_data)

        # if self.prfx:
        #     max_d = len(train_data[0][1]) - 1
        #     x_train_len = np.array([len(x) for x in x_train])
        # x_train_len = np.repeat([len(x) for x in x_train], max_d + 1)
        # x_valid_len = np.repeat([len(x) for x in x_valid], max_d + 1)
        # x_test_len = np.repeat([len(x) for x in x_test], max_d + 1)
        dl_train = self.dataloader(x_train, y_train, shuffle=True)
        x_train_len = []
        for batch in dl_train:
            lengths = batch[0][-1].cpu().numpy()
            x_train_len.extend(lengths)

        if valid_data:
            dl_valid = self.dataloader(x_valid, y_valid, shuffle=False)
        if test_data:
            dl_test = self.dataloader(x_test, y_test, shuffle=False)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # --- initialize optimizer --- #
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=self.l2)
        if is_swa:
            opt = SWA(opt, swa_start=5, swa_freq=2, swa_lr=lr / 2)

        # --- create summary writer --- #
        sw = SummaryWriter(logdir)

        global_step = 0  # increased by one whenever a batch is generated

        q_error_bsf = 1000000
        min_epoch = 10
        if self.n_char <= 10:
            min_epoch = 40
        patience = 0
        max_patience = self.patience

        for epoch in range(1, max_epoch + 1):
            # --- training loop --- #
            tqdm_train = tqdm(dl_train, total=len(dl_train), mininterval=5)
            for batch in tqdm_train:
                # print(f"batch start {global_step}")

                # debug hook
                # if global_step >= 10:
                #     break

                global_step += 1

                opt.zero_grad()

                # --- send data to device (CPU or GPU) --- #
                x_batch, y_batch = self._device_batch(batch)
                # print(f"============ start epoch : {epoch} {global_step} =============")
                # print(x_batch, y_batch)
                # print(f"============ end epoch : {epoch} {global_step} =============")

                loss_batch = model.loss(x_batch, y_batch)
                loss_batch.backward()
                if self.clip_gr > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_gr)

                opt.step()

                # evaluate for train_batch
                loss_batch_ = loss_batch.item()
                # try:
                #     loss_batch_ = loss_batch.item()
                #     print(f"============ start in try epoch : {epoch} {global_step} =============")
                #     print(x_batch, y_batch)
                #     print(f"============ end in try epoch : {epoch} {global_step} =============")
                # except RuntimeError as e:
                #     print(batch)
                #     traceback.print_exc()

                x_batch, y_batch = self._device_batch(batch)
                pred_batch = model.forward(x_batch, prfx_train=self.seq_out)
                pred_batch = pred_batch.detach().cpu().numpy()
                y_batch = y_batch.cpu().numpy()
                if self.seq_out:
                    if self.prfx:
                        x_batch_len = x_batch[-1].cpu().numpy()
                        q_error_batch = self.q_error(y_batch, pred_batch, x_batch_len)
                    else:
                        q_error_batch = self.q_error(y_batch, pred_batch)
                    # y_batch_last = np.take_along_axis(y_batch, np.expand_dims(x_batch_len, 1)-1, 1).squeeze()
                    # q_error_batch = self.q_error(y_batch_last, pred_batch)
                else:
                    q_error_batch = self.q_error(y_batch, pred_batch)

                # write stats with std and log
                tqdm_train.set_description(f"train_loss: {loss_batch_:07.3f} train_q_error: {q_error_batch:07.3f}",
                                           refresh=False)  # tqdm print setting
                if global_step % 1000 == 0:
                    sw.add_scalars("Batch", {"q_err": q_error_batch, "loss": loss_batch_},
                                   global_step=global_step)  # add stats

            # --- validation --- #
            if is_swa:  # To evaluate model
                opt.swap_swa_sgd()

            if is_debug:
                pred_train, y_train = self.estimate(dl_train)  # for train
                if self.prfx:
                    q_error_train = self.q_error(y_train, pred_train, x_train_len)
                    # y_train_last = np.take_along_axis(y_train, np.expand_dims(x_train_len, 1) - 1, 1).squeeze()
                    # q_error_train = self.q_error(y_train_last, pred_train)
                else:
                    q_error_train = self.q_error(y_train, pred_train)
                loss_train = self._average_loss(dl_train)
            else:
                loss_train = -1
                q_error_train = -1

            pred_valid, y_valid = self.estimate(dl_valid)  # for valid
            q_error_valid = self.q_error(y_valid, pred_valid)
            loss_valid = self._average_loss(dl_valid)

            if test_data:
                pred_test, y_test = self.estimate(dl_test)  # for test
            # if self.prfx:
            #     q_error_test = self.q_error(y_test, pred_test, x_test_len)
            # else:
            #     q_error_test = self.q_error(y_test, pred_test)
                q_error_test = self.q_error(y_test, pred_test)
                loss_test = self._average_loss(dl_test)
            else:
                loss_test = -1
                q_error_test = -1

            sw.add_scalars("Epoch_loss", {"train": loss_train, "valid": loss_valid, "test": loss_test},
                           global_step=epoch)  # add stats
            sw.add_scalars("Epoch_q_err", {"train": q_error_train, "valid": q_error_valid, "test": q_error_test},
                           global_step=epoch)  # add stats
            sw.flush()
            print(f"[epoch {epoch:02d}] valid_loss: {loss_valid:07.3f}, valid_q_error: {q_error_valid:07.3f}")
            if test_data:
                print(f"[epoch {epoch:02d}] test_loss: {loss_test:07.3f}, test_q_error: {q_error_test:07.3f}")

            if q_error_bsf > q_error_valid:
                q_error_bsf = q_error_valid
                torch.save(model.state_dict(), curr_best_path)
                patience = 0
                self.best_epoch = epoch
            else:
                patience += 1

            if epoch <= min_epoch:
                patience = 0

            self.last_epoch = epoch
            if patience == max_patience:
                break

            if is_swa:  # To continue training
                opt.swap_swa_sgd()
        os.rename(curr_best_path, save_path)
        self.model.load_state_dict(torch.load(save_path))
        torch.save(self.model.state_dict(), save_path)

    def _average_loss(self, data_or_loader):
        model = self.model
        if isinstance(data_or_loader, DataLoader):
            dl_data = data_or_loader  # data_loader
        else:
            x_data, y_data = self._encode_data(data_or_loader)  # data
            dl_data = self.dataloader(x_data, y_data, shuffle=False)

        loss = 0
        n_data = 0
        for batch in dl_data:
            # --- send data to device (CPU or GPU) --- #
            x_batch, y_batch = self._device_batch(batch)
            l_data = len(y_batch)
            loss_batch = model.loss(x_batch, y_batch).item()
            loss += loss_batch * l_data
            n_data += l_data
        loss /= n_data
        return loss

    def query(self, str_q, d):
        queries = [(str_q, d, 0)]
        result = self.estimate(queries)
        return result[0][0]

    def estimate_latency_anlysis(self, test_data):
        return super().estimate_latency_anlysis(test_data)

    def estimate(self, data_or_loader):
        dconf = self.db_factory.conf
        n = dconf.n

        model = self.model
        model.eval()  # evaluation mode

        if isinstance(data_or_loader, DataLoader):
            dl_data = data_or_loader  # data_loader
        else:
            x_data, y_data = self._encode_data(data_or_loader)  # data
            dl_data = self.dataloader(x_data, y_data, shuffle=False)
        y_pred = []
        y_data = []
        for batch in dl_data:
            # --- send data to device (CPU or GPU) --- #
            y_batch = batch[1].numpy()  # CPU Tensor data
            prfx_train = (self.seq_out is True) and isinstance(y_batch[0], Iterable)
            x_batch, _ = self._device_batch(batch)  # device data
            pred_batch = model.forward(x_batch, prfx_train=prfx_train)

            pred_batch = pred_batch.detach().cpu().numpy()  # send results to main memory
            y_pred.append(pred_batch)
            y_data.extend(y_batch)

        if (self.seq_out is True) and isinstance(y_batch[0], Iterable):
            max_len = max([x.shape[-1] for x in y_pred])
            y_pred_bak = y_pred
            y_pred = []
            y_data_bak = y_data
            y_data = []
            for pred_batch in y_pred_bak:
                pred_batch_pad = np.zeros((pred_batch.shape[0], max_len))
                pred_batch_pad[:pred_batch.shape[0], :pred_batch.shape[1]] = pred_batch
                y_pred.append(pred_batch_pad)

            for y_batch in y_data_bak:
                y_batch_pad = np.zeros(max_len)
                y_batch_pad[:y_batch.shape[0]] = y_batch
                y_data.append(y_batch_pad)

            y_pred = np.vstack(y_pred)
            y_data = np.vstack(y_data)

        else:
            y_pred = np.hstack(y_pred)

        y_pred[y_pred < 0.] = 0.
        y_pred[y_pred > n] = n
        model.train()  # training mode
        return y_pred, y_data


class MSLELoss(nn.Module):
    def __init__(self, reduction=None):
        super().__init__()
        if reduction is None:
            reduction = 'mean'
        self.reduction = reduction

    def forward(self, pred, actual):
        epsilon = 1e-7
        y_pred = pred
        y_true = actual
        first_log = torch.log(torch.clamp_min(y_pred, epsilon) + 1)
        # ---- print predicted values --- #
        # print(y_pred)
        # print(torch.clamp_min(y_pred, epsilon))
        # print(torch.clamp_min(y_pred, epsilon) + 1)
        # print(first_log)
        # exit()
        # ---- end predicted values --- #
        second_log = torch.log(torch.clamp_min(y_true, epsilon) + 1)
        return F.mse_loss(first_log, second_log, reduction=self.reduction)


def mean_squared_logarithmic_loss(pred, actual, lengths=None, reduction='mean', bt_mode=False):
    """

    :param pred:
    :param actual:
    :param lengths: for masking
    :param reduction:
    :return:
    """
    loss_func = MSLELoss(reduction)
    loss = loss_func(pred, actual)
    if lengths is not None:
        assert reduction == 'none', "reduction should not be None"
        assert len(loss) == len(lengths), "length consistency fails"
        max_len = max(lengths)
        # mask = np.arange(max_len).reshape(1,-1).repeat(len(lengths),0) # np
        if bt_mode:
            mask = (actual > 0)
        else:
            mask = torch.arange(max_len).repeat(len(lengths), 1).to(loss.device)
            mask = mask < lengths.reshape(-1, 1)
        mask = mask.to(loss.device)
        loss = mask * loss
    return loss


def weighted_mean_squared_logarithmic_loss(pred, actual, weight, reduction=None):
    if reduction is None:
        reduction = 'mean'
    loss_func = MSLELoss(reduction='none')
    output = weight * loss_func(pred, actual)
    if reduction == "mean":
        output = torch.mean(output)
    elif reduction == "sum":
        output = torch.sum(output)
    elif reduction == "none":
        output = output  # pass output
    else:
        raise ValueError(f"reduction option [{reduction}] is not supported")

    return output


@torch.no_grad()
def init_weights(model):
    if type(model) == nn.Linear:
        torch.nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(0.01)


@torch.no_grad()
def init_weights_vae(model):
    constant = 0.01
    if type(model) == nn.Linear:
        # torch.nn.init.kaiming_uniform_(model.weight)
        torch.nn.init.uniform_(model.weight, -constant, constant)

        if model.bias is not None:
            model.bias.data.fill_(0.0)


class ConcatEmbed(nn.Module):
    def __init__(self, n_char, delta, emb_size, dist_emb_size=5):
        super().__init__()
        self.char_embedding = nn.Embedding(n_char + 1, emb_size - dist_emb_size,
                                           padding_idx=0)  # padding_idx currently not working
        self.dist_embedding = nn.Embedding(delta + 1, dist_emb_size)

    def forward(self, x, d):
        char_emb = self.char_embedding(x)  # batch, max_len, emb-dist_emb
        batch_size = char_emb.shape[0]
        max_len = char_emb.shape[1]
        d_emb = self.dist_embedding(d)  # batch, dist_emb
        d_emb = d_emb.repeat(1, max_len)  # batch, max_len * dist_emb
        d_emb = d_emb.reshape(batch_size, max_len, -1)  # batch, max_len, dist_emb

        output = torch.cat([char_emb, d_emb], dim=-1)  # batch, max_len, emb_size
        return output


class RNN_module(nn.Module):
    def __init__(self, n_char, pred_hs, rnn_hs, embed_size, num_pred_layer=5,  max_d=3,
                 num_rnn_layer=1, prfx=False, max_len=None, n_channel=1, btS=None, btA=None):
        super().__init__()
        self.num_rnn_layer = num_rnn_layer
        self.prfx = prfx
        self.btS = btS
        self.btA = btA
        self.bt_mode = btS or btA
        self.seq_out = self.prfx or self.btS or self.btA
        self.max_len = max_len

        # input_size = n_char+1
        # self.char_embedding = nn.Embedding(input_size, embed_size - 5, padding_idx=0)
        # self.dist_embedding = nn.Embedding(max_d+1, 5)
        # nn.init.uniform_(self.char_embedding.weight)
        # nn.init.uniform_(self.dist_embedding.weight)
        dist_emb_size = 5
        assert embed_size > dist_emb_size
        self.embedding = ConcatEmbed(n_char, max_d, embed_size, dist_emb_size=dist_emb_size)
        # nn.init.uniform_(self.embedding.char_embedding.weight)
        # nn.init.uniform_(self.embedding.dist_embedding.weight)
        nn.init.xavier_normal_(self.embedding.char_embedding.weight)
        nn.init.xavier_normal_(self.embedding.dist_embedding.weight)

        assert rnn_hs % 2 == 0, f"rnn hidden size should be even, but {rnn_hs} are given"
        self.rnn_hs = rnn_hs

        cell_size = rnn_hs
        self.rnns = nn.ModuleList()
        self.rnn = nn.LSTM(embed_size, cell_size, batch_first=True, num_layers=num_rnn_layer)
        self.rnns.append(self.rnn)
        for rnn in self.rnns:
            for name, param in rnn.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
                elif "weight_ih_l0" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh_l0":
                    nn.init.orthogonal_(param)
                else:
                    raise ValueError("parameter is not initialized")

        self.pred = nn.Sequential()
        for id in range(num_pred_layer):
            id += 1
            if id == 1:
                self.pred.add_module(f"PRED-{id}", nn.Linear(rnn_hs, pred_hs))
            else:
                self.pred.add_module(f"PRED-{id}", nn.Linear(pred_hs, pred_hs))
            self.pred.add_module(f"LeakyReLU-{id}", nn.LeakyReLU())
        self.pred.add_module(f"PRED-OUT", nn.Linear(pred_hs, n_channel))
        self.pred.apply(init_weights)

    def logit(self, x, hidden=None, prfx_train=False):
        data, delta, lengths = x
        embed = self.embedding(data, delta)
        packed = pack_padded_sequence(embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # rnn = self.rnn
        output, hidden = self.rnn(packed, hx=hidden)
        if self.seq_out:
            hidden = pad_packed_sequence(output)[0].transpose(0, 1)
        else:
            if isinstance(self.rnn, torch.nn.LSTM):
                hidden, cell = hidden
                hidden = torch.transpose(hidden, 0, 1)
                # hidden = hidden.reshape(-1, self.rnn_hs)
        if not self.seq_out:
            hidden = hidden[:, -1, :]  # select last output (cf., include the case when single layered rnn)

        # if self.prfx:
        #     max_batch_len = hidden.size()[1]
        #     output = []
        #     for pos in range(max_batch_len):
        #         output.append(self.preds[min(pos, self.max_len-1)](hidden[:, pos, :]))
        #     output = torch.stack(output, dim=1)
        # else:
        #     output = self.pred(hidden)
        output = self.pred(hidden)

        output = torch.mean(output, dim=-1)
        # print(output)
        # if len(output) > 1:
        #     exit()

        # print("hidden:", hidden)
        # print("before last:", output)
        # output = self.last(output)
        # print("after last:", output)
        # if self.prfx:
        if self.seq_out and not prfx_train:
            output = torch.gather(output, 1, (lengths - 1).unsqueeze(-1).cuda()).squeeze()
        return output

        # if hidden is not None:
        #     hidden = torch.from_numpy(np.array(hidden)).float()
        # else:
        #     if isinstance(self.rnn, nn.LSTM):
        #         h0 = torch.zeros(1, len(x), self.unit_size)
        #         c0 = torch.zeros(1, len(x), self.unit_size)
        #         hidden = (h0, c0)
        #     elif isinstance(self.rnn, nn.RNN):
        #         hidden = torch.zeros(1, len(x), self.unit_size)
        # x = torch.from_numpy(np.array(x))
        # x = self.embedding(x)
        # out, hidden = self.rnn(x)
        # out = out[:, -1, :]
        # for i, layer in enumerate(self.layers):
        #     out = layer(out)
        # return out

    def forward(self, x, hidden=None, prfx_train=False):
        out = self.logit(x, prfx_train=prfx_train)
        out = F.leaky_relu(out)
        out = torch.squeeze(out)
        return out

    def loss(self, x, y):
        # assert isinstance(y, torch.Tensor)
        # prfx_train = (self.prfx is True) and y[0].dim() > 0
        seq_train = (self.seq_out is True) and y[0].dim() > 0
        pred_y = self.forward(x, prfx_train=seq_train)
        # if self.prfx:
        #     y = y.reshape(-1, y.size()[-1])
        # print("pred:", pred_y)
        # exit()
        # loss = F.smooth_l1_loss(pred_y, y)

        if seq_train:
            data, delta, lengths = x
            loss = mean_squared_logarithmic_loss(pred_y, y, lengths, bt_mode=self.bt_mode, reduction='none')
            if self.prfx:
                # y_last = torch.gather(y, 1, (lengths - 1).unsqueeze(-1).cuda()).squeeze()
                # loss = mean_squared_logarithmic_loss(pred_y, y_last)
                loss = loss.sum() / lengths.sum()
            else:
                # loss = mean_squared_logarithmic_loss(pred_y, y, lengths, reduction='none')
                loss = loss.sum() / torch.gt(y, 0).sum()
        else:
            loss = mean_squared_logarithmic_loss(pred_y, y)
        return loss


def encode_string_by_CardNet(string, sigma, tau_max, l_max):
    bin_len = (l_max + 2 * tau_max) * (len(sigma) + 1)  # to support [UNK] token
    bin = np.zeros(bin_len, dtype=np.int)
    for pos, ch in enumerate(string):
        ch_id = 1  # initialize [UNK]
        if ch in sigma:
            ch_id = sigma[ch]
        ch_id -= 1  # shift index to ignore [PAD] token
        start_pos = ch_id * (l_max + 2 * tau_max) + pos
        bin[start_pos: start_pos + 2 * tau_max + 1] = 1
    return bin


class Elemwise_Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.bias = None
        torch.nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x):
        x = torch.sum(torch.mul(x, self.weight), dim=-1) + self.bias
        if self.bias is not None:
            x += self.bias
        return x


class VAE_CardNet(nn.Module):
    def __init__(self, input_size, vclip_lv, z_size=128, scale=None):
        super().__init__()
        if scale is None:
            scale = 128
        self.z_size = z_size
        self.scale = scale
        self.enc_hiddens = [input_size, scale * 2, scale, scale, z_size * 2]  # last hidden for mu and sigma
        self.dec_hiddens = [z_size, scale, scale, scale * 2, input_size]
        self.vclip_lv = vclip_lv

        # ---- ENCODER VAE
        self.encoder = nn.Sequential()
        prev_hidden = self.enc_hiddens[0]
        for i, hidden in enumerate(self.enc_hiddens[1:]):
            l_id = i + 1
            if i:
                self.encoder.add_module(f"ENC_VAE_ACT{l_id-1}", nn.ELU())
            self.encoder.add_module(f"ENC_VAE{l_id}", nn.Linear(prev_hidden, hidden))
            prev_hidden = hidden
        self.encoder.apply(init_weights_vae)

        # ---- DECODER VAE
        self.decoder = nn.Sequential()
        prev_hidden = self.dec_hiddens[0]
        for i, hidden in enumerate(self.dec_hiddens[1:]):
            l_id = i + 1
            self.decoder.add_module(f"DEC_VAE{l_id}", nn.Linear(prev_hidden, hidden))
            if l_id < len(self.dec_hiddens[1:]):
                # self.decoder.add_module(f"DEC_VAE{l_id}", nn.Linear(prev_hidden, hidden))
                self.decoder.add_module(f"DEC_VAE_ACT{l_id}", nn.ELU())
            else:
                pass
                # self.decoder.add_module(f"DEC_VAE{l_id}", nn.Linear(prev_hidden, hidden, bias=False))
            #     self.decoder.add_module(f"DEC_VAE_ACT{l_id}", nn.Sigmoid())
            prev_hidden = hidden
        # self.decoder.add_module("DEC_VAE1", nn.Linear(z_size, 128))
        # self.decoder.add_module("DEC_VAE_ACT1", nn.ELU())
        # self.decoder.add_module("DEC_VAE2", nn.Linear(128, 128))
        # self.decoder.add_module("DEC_VAE_ACT2", nn.ELU())
        # self.decoder.add_module("DEC_VAE3", nn.Linear(128, 256))
        # self.decoder.add_module("DEC_VAE_ACT3", nn.ELU())
        # self.decoder.add_module("DEC_VAE4", nn.Linear(256, input_size))
        # self.decoder.add_module("DEC_VAE_ACT4", nn.Sigmoid())
        self.decoder.apply(init_weights_vae)

    def encode(self, x):
        return self.reparameterize(*self.encode_param(x))

    def encode_param(self, x):
        mu, logvar = self.encoder(x).split(self.z_size, dim=-1)
        # soft value clipping
        scale = self.vclip_lv
        if scale > 0:
            logvar = torch.tanh(logvar / scale) * scale
        return mu, logvar

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return (eps * std) + mu
        else:
            return mu

    def _forward(self, x):
        mu, logvar = self.encode_param(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def forward(self, x):
        recon_x, _, _ = self._forward(x)
        return recon_x

    def loss(self, x, output_both=False):
        mu, logvar = self.encode_param(x)  # if the value of logvar is larger than 100, it makes infinity error.
        if is_debug:
            ut.check_nan_inf(mu)
        if is_debug:
            ut.check_nan_inf(logvar)
        if is_debug:
            var = logvar.exp()
            ut.check_nan_inf(var)

        z = self.reparameterize(mu, logvar)
        if is_debug:
            ut.check_nan_inf(z)

        recon_x_logits = self.decoder(z)
        if is_debug:
            ut.check_nan_inf(recon_x_logits)

        # ---- Reconstruction loss
        # BCE2 = F.binary_cross_entropy(recon_x, x, reduction='sum')
        bce_loss = F.binary_cross_entropy_with_logits(recon_x_logits, x, reduction='sum')
        if is_debug:
            ut.check_nan_inf(bce_loss)

        # ---- Regularize loss (normal distribution)
        # kld_loss = torch.mul(torch.sum(1 + logvar - mu.pow(2) - logvar.exp()), -0.5)
        kld_loss = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())
        if is_debug:
            ut.check_nan_inf(kld_loss)
        output_loss = bce_loss + kld_loss

        # normalized by the corresponding vector size
        bce_loss /= recon_x_logits.size()[-1]
        kld_loss /= mu.size()[-1]
        if output_both:
            return output_loss, bce_loss,  kld_loss
        else:
            return output_loss


class CardNet(nn.Module):
    def __init__(self, l_max, n_char, vclip_lv, tau_max=None, scale=None, vscale=None, z_size=None, learn_sep=False, device="cuda"):
        super().__init__()
        if tau_max is None:
            tau_max = 3
        if scale is None:
            scale = 256
        if z_size is None:
            z_size = 128

        # self.theta_max = theta_max
        if learn_sep:
            self.tau_max = 0
        else:
            self.tau_max = tau_max
        # print("[Init CardNet]", "learn_sep:", learn_sep, "self.tau_max:", self.tau_max)
        self.l_max = l_max
        # assert tau_max == theta_max, "It does not support different tau and theta"
        self.dynamic_train = True
        # print("l_max:", l_max, "tau_max:", tau_max, "n_char:", n_char)
        input_size = (l_max + 2 * tau_max) * (n_char + 1)  # n_char include the [UNK] token
        dist_emb_size = 5
        # ---- VAE
        self.vae: VAE_CardNet = VAE_CardNet(input_size, vclip_lv, z_size=z_size, scale=vscale)

        # ---- Distance embedding
        self.dist_emb = nn.Embedding(self.tau_max + 1, dist_emb_size)
        nn.init.normal_(self.dist_emb.weight)

        # ---- Final Encoder Phi
        self.enc_final = nn.Sequential()
        self.scale = scale
        combine_size = input_size + z_size + dist_emb_size
        last_hidden = 60  # z_x^i
        self.enc_hiddens = [combine_size, scale*2, scale*2, scale, scale, last_hidden]  # last hidden for mu and sigma

        prev_hidden = self.enc_hiddens[0]
        for i, hidden in enumerate(self.enc_hiddens[1:]):
            l_id = i+1
            if i:
                self.enc_final.add_module(f"ENC_FIN_ACT{l_id-1}", nn.ReLU())
            self.enc_final.add_module(f"ENC_FIN{l_id}", nn.Linear(prev_hidden, hidden))
            prev_hidden = hidden

        # self.enc_final.add_module("ENC_FIN1", nn.Linear(combine_size, 512))
        # self.enc_final.add_module("ENC_FIN_ACT1", nn.ReLU())
        # self.enc_final.add_module("ENC_FIN2", nn.Linear(512, 512))
        # self.enc_final.add_module("ENC_FIN_ACT2", nn.ReLU())
        # self.enc_final.add_module("ENC_FIN3", nn.Linear(512, 256))
        # self.enc_final.add_module("ENC_FIN_ACT3", nn.ReLU())
        # self.enc_final.add_module("ENC_FIN4", nn.Linear(256, 256))
        # self.enc_final.add_module("ENC_FIN_ACT4", nn.ReLU())
        # self.enc_final.add_module("ENC_FIN_OUT", nn.Linear(256, 60))
        self.enc_final.apply(init_weights_vae)

        # ---- Final Decoders (tau_max)
        self.decoders = nn.Sequential()
        decoder_linear = Elemwise_Linear(last_hidden, self.tau_max + 1)
        nn.init.kaiming_uniform_(decoder_linear.weight, nonlinearity="relu")
        decoder_linear.bias.data.fill_(0.0)
        self.decoders.add_module("DEC_FIN_OUT", decoder_linear)
        self.decoders.add_module("DEC_FIN_ACT", nn.ReLU())

        self.default_dynamic_weight = torch.ones(self.tau_max + 1, requires_grad=False) / (self.tau_max + 1)
        self.default_dynamic_weight = self.default_dynamic_weight.to(device)
        self.dynamic_weight = self.default_dynamic_weight
        self.prev_val_loss = None

    def update_dynamic_weight(self, val_loss) -> None:
        """
            update dynamic_weight and previous validation loss
        :param val_loss:
        :return:
        """
        if self.prev_val_loss is None:
            dynamic_weight = self.default_dynamic_weight
        else:
            assert len(val_loss) == self.tau_max + 1
            assert len(self.prev_val_loss) == self.tau_max + 1
            val_loss_diff = val_loss - self.prev_val_loss

            val_loss_diff[val_loss_diff < 0] = 0
            pos_sum = sum(val_loss_diff)
            if pos_sum > 0:
                dynamic_weight = val_loss_diff / sum(val_loss_diff)
            else:  # all losses decrease
                dynamic_weight = self.default_dynamic_weight

        # assign dynamic_weight
        dynamic_weight = dynamic_weight.detach()
        print("dynamic weights are updated from", self.dynamic_weight, "to", dynamic_weight)
        self.dynamic_weight = dynamic_weight

        # setting previous validation loss
        self.prev_val_loss = val_loss

    def dynamic_loss(self, c_hat, c, reduction=None):
        if reduction is None:
            reduction = "mean"
        weight = self.dynamic_weight
        dynamic_loss = weighted_mean_squared_logarithmic_loss(c_hat, c, weight, reduction=reduction)
        return dynamic_loss

    def pred_loss(self, x, c, reduction=None):
        c_hat = self.forward(x)
        loss_g = mean_squared_logarithmic_loss(c_hat, c, reduction=reduction)
        if self.dynamic_train:
            dynamic_lambda = 0.1  # according to the CardNet paper
            loss_g += dynamic_lambda * self.dynamic_loss(c_hat, c)
        return loss_g

    def loss(self, x, c):
        vae_lambda = 0.1  # according to the CardNet paper
        return self.pred_loss(x, c) + vae_lambda * self.vae.loss(x)

    def forward(self, x, d=None, mono=True):
        """

        :param x:
        :param d:
        :param mono: cumulative sum for ensuring monotinicity
        :return:
        """
        if d is None:
            z = self.vae.encode(x)
            x_prime = torch.cat([x, z], dim=-1)
            # print("tau_max in forward of CardNet:", self.tau_max)
            x_prime = x_prime.unsqueeze(1).repeat([1, self.tau_max + 1, 1])
            batch_size = len(x)
            # print("dist_emb.weight size:", self.dist_emb.weight.size())
            dist_emb_repeat = self.dist_emb.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            # print("x_prime size:", x_prime.size())
            # print("dist_emb_repeat size:", dist_emb_repeat.size())
            x_d = torch.cat([x_prime, dist_emb_repeat], dim=-1)
            x_d = x_d.reshape(-1, x_d.size(-1))
            z_d = self.enc_final(x_d)
            z_d = z_d.reshape(batch_size, self.tau_max + 1, -1)
            output = self.decoders(z_d)

            if mono:
                output = torch.cumsum(output, 1)
        else:
            output = self.forward(x, mono=mono)
            output = output[:, d]
        return output


class CardNetDataset(Dataset):
    def __init__(self, data, sigma, tau_max, l_max):
        super().__init__()
        self.data = data
        self.sigma = sigma
        self.tau_max = tau_max
        self.l_max = l_max

    def __getitem__(self, item):
        data = self.data[item]
        # string, c = data[0], data[1:]
        string, c = data
        x = encode_string_by_CardNet(string, self.sigma, self.tau_max, self.l_max)

        return x, c

    def __len__(self):
        return len(self.data)


def cardnet_vae_agg_(items):
    x, c = list(zip(*items))
    return torch.FloatTensor(x)


def cardnet_agg_(items):
    x, c = list(zip(*items))
    return torch.FloatTensor(x), torch.FloatTensor(c)


class CardNetEstimator(Estimator):
    def __init__(self, conf):
        super().__init__(conf)
        self.class_name = ut.get_class_name(self)
        self.seed = conf.seed
        self.lr = conf.lr
        self.vlr = conf.vlr
        self.l2 = conf.l2
        self.vl2 = conf.vl2
        self.bs = conf.bs
        self.vbs = conf.vbs
        self.patience = conf.patience
        self.max_epoch_vae = conf.max_epoch_vae
        self.max_epoch = conf.max_epoch
        self.is_swa = conf.swa
        self.csc = self.conf.csc  # CardNet scale
        self.vsc = self.conf.vsc  # VAE scale
        self.tau_max = conf.max_d
        self.delta = conf.delta
        if self.delta is not None:
            self.tau_max = 0
        self.max_char = self.conf.max_char
        self.vclip_lv = self.conf.vclip_lv
        self.vclip_gr = self.conf.vclip_gr
        self.clip_gr = self.conf.clip_gr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.ncores = self.conf.ncores
        self.save_dir = None
        self.save_path = None
        self.log_dir = None
        self.best_epoch = 0
        self.last_epoch = 0
        self.best_epoch_vae = 0
        self.last_epoch_vae = 0

    def _dataset(self, data):
        if self.delta is not None:
            dataset = CardNetDataset(data, self.sigma, self.delta, self.l_max)
        else:
            dataset = CardNetDataset(data, self.sigma, self.tau_max, self.l_max)
        return dataset

    def _dataloader(self, dataset, shuffle):
        if not isinstance(dataset, Dataset):
            dataset = self._dataset(dataset)
        dl = DataLoader(dataset, batch_size=self.bs, shuffle=shuffle, collate_fn=cardnet_agg_)
        return dl

    def _dataloader_vae(self, dataset, shuffle):
        # print("dataset in dataload:", dataset[:2])
        # exit()
        if not isinstance(dataset, Dataset):
            dataset = self._dataset(dataset)
        dl = DataLoader(dataset, batch_size=self.vbs, shuffle=shuffle, collate_fn=cardnet_vae_agg_)
        return dl

    # def _featurizer(self, input_string, sigma, tau_max, l_max):
    #     return encode_string_by_CardNet(input_string, sigma, tau_max, l_max)

    def build(self, train_data, valid_data=None, test_data=None, over_write=False):
        db = self.db_factory.get_db()
        self.sigma = ut.char_dict_from_db(db, self.max_char)
        # self.l_max = max([len(x) for x in db])
        self.l_max = max([len(x[0]) for x in train_data])
        n_char = len(self.sigma)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.delta is not None:
            self.tau_max = self.delta
        self.model: CardNet = CardNet(self.l_max, n_char, self.vclip_lv, tau_max=self.tau_max, scale=self.csc,
                                      vscale=self.vsc, learn_sep=self.delta is not None, device=self.device)
        self.model.to(self.device)
        if self.delta is not None:
            self.tau_max = 0
        ut.print_torch_summarize(self.model)

        assert self.save_dir is not None, "save dir should be given"
        assert self.save_path is not None, "save path should be given"
        assert self.logdir is not None, "log dir should be given"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.logdir, exist_ok=True)
        self.vae_path = self.save_dir + "/vae.pt"

        # load or train model
        if not over_write and os.path.exists(self.save_path):
            self.model.load_state_dict(torch.load(self.save_path, map_location=torch.device(self.device)))
            self.model.to(self.device)
        else:
            # load or train vae
            if not over_write and os.path.exists(self.vae_path):
                self.model.vae.load_state_dict(torch.load(self.vae_path, map_location=torch.device(self.device)))
            else:
                self._train_vae(train_data, valid_data)
            self._train_cardnet(train_data, valid_data, test_data)

    def model_size(self, *args):
        # vae_size = os.path.getsize(self.vae_path)
        est_size = os.path.getsize(self.save_path)  # CardNet includes vae_size
        return est_size

    def _batch_to_device(self, batch):
        x, c = batch
        x = x.to(self.device)
        c = c.to(self.device)
        return x, c

    def eval_cardnet(self, data_or_dataloader, update_dynamic_weight=False):
        self.model.eval()
        if isinstance(data_or_dataloader, DataLoader):
            dataloader = data_or_dataloader
        else:
            dataloader = self._dataloader(data_or_dataloader, shuffle=False)

        loss = 0
        total_n = 0
        q_error = 0
        loss_each = torch.zeros(self.tau_max + 1, requires_grad=False).to(self.device)
        for batch in dataloader:
            x, c = self._batch_to_device(batch)
            c_hat = self.model.forward(x)
            c_hat = c_hat.detach()  # delete grad part
            # batch_loss = self.model.loss(x, c)
            batch_loss_each = torch.mean(mean_squared_logarithmic_loss(c_hat, c, reduction='none'), dim=0)
            batch_loss = torch.mean(batch_loss_each)
            batch_q_error = ut.q_error(c.cpu().numpy().reshape(-1), c_hat.cpu().numpy().reshape(-1))
            batch_n = len(c)
            loss += batch_loss.item()
            loss_each += batch_loss_each * batch_n
            q_error += batch_q_error * batch_n

            total_n += batch_n
        loss /= total_n
        loss_each /= total_n
        q_error /= total_n

        if update_dynamic_weight:
            self.model.update_dynamic_weight(val_loss=loss_each)

        self.model.train()
        return loss, q_error

    def eval_vae(self, data_or_dataloader):
        self.model.vae.eval()
        if isinstance(data_or_dataloader, DataLoader):
            dataloader = data_or_dataloader
        else:
            dataloader = self._dataloader_vae(data_or_dataloader, shuffle=False)

        loss = 0
        total_n = 0
        for batch in dataloader:
            # x, c = self._batch_to_device(batch)
            batch_n = len(batch)
            x = batch.to(self.device)
            batch_loss = self.model.vae.loss(x)
            loss += batch_loss.cpu().item()
            total_n += batch_n
        loss /= total_n
        self.model.vae.train()
        return loss

    def estimate_latency_anlysis(self, test_data):
        return super().estimate_latency_anlysis(test_data)

    def estimate(self, test_data):
        self.model.eval()
        n = self.db_factory.conf.n
        dataloader = self._dataloader(test_data, shuffle=False)

        y_true = []
        y_pred = []
        for batch in dataloader:
            x, c = self._batch_to_device(batch)
            y_true.extend(c.cpu().numpy().reshape(-1))
            c_hat = self.model.forward(x)
            y_pred.extend(c_hat.detach().cpu().numpy().reshape(-1))
        y_pred = np.array(y_pred)
        y_pred[y_pred < 0.] = 0.
        y_pred[y_pred > n] = n
        self.model.train()
        return y_pred, y_true

    def _train_vae(self, train_data, valid_data=None):
        self.model.vae.to(self.device)
        self.model.vae.train()
        dl_train = self._dataloader_vae(train_data, shuffle=True)
        dl_valid = self._dataloader_vae(valid_data, shuffle=False)

        # ---- VAE training
        opt_vae = optim.Adam(self.model.vae.parameters(), lr=self.vlr, weight_decay=self.vl2)
        print("vlr:", self.vlr, "weight decay:", self.vl2, "vclip logvar:", self.vclip_lv, "vclip grad:", self.vclip_gr)
        if self.is_swa:
            opt_vae = SWA(opt_vae, swa_start=5, swa_freq=2, swa_lr=self.vlr / 2)

        min_epoch = 10
        max_patience = self.patience
        curr_patience = 0
        curr_min_loss = 10000
        self.vae_path_curr = self.save_dir + "/_best_vae.pt"
        sw = SummaryWriter(self.logdir)
        if is_debug:
            logdir_debug = "debug_log/"
            os.makedirs(logdir_debug, exist_ok=True)
            count_debug = len(os.listdir(logdir_debug)) + 1
            sw_debug = SummaryWriter(
                logdir_debug + f"{count_debug}_vlr_{self.vlr}_vl2_{self.vl2}_vclipLv_{self.vclip_lv}_vclipGr_{self.vclip_gr}/")
        step = 0
        torch.save(self.model.vae.state_dict(), self.vae_path_curr)
        for epoch in range(1, self.max_epoch_vae + 1):
            tqdm_train = tqdm(dl_train, total=len(dl_train), mininterval=5)
            # ---- train loop
            self.model.vae.train()
            for batch in tqdm_train:
                opt_vae.zero_grad()
                step += 1
                x = batch.to(self.device)
                # x, c = self._batch_to_device(batch)
                # print("x:", x.size())
                # print("model:")
                # print(ut.torch_summarize(self.model.vae))
                loss = self.model.vae.loss(x)
                if is_debug:
                    loss, bce_loss, kld_loss = self.model.vae.loss(x, output_both=True)
                    loss_val = float(loss) / len(batch)
                    bce_val = float(bce_loss) / len(batch)
                    kld_val = float(kld_loss) / len(batch)
                    sw_debug.add_scalar("VAE_loss", loss_val, global_step=step)
                    sw_debug.add_scalar("VAE_BCE", bce_val, global_step=step)
                    sw_debug.add_scalar("VAE_KLD", kld_val, global_step=step)
                loss.backward()
                if self.vclip_gr > 0:
                    torch.nn.utils.clip_grad_value_(self.model.vae.parameters(), self.vclip_gr)
                if is_debug:
                    is_grad_explode = ut.is_grad_explode_in_model(self.model.vae)
                    if is_grad_explode:
                        for name, param in self.model.vae.named_parameters():
                            print(name, param)
                            print("grad:", param.grad)
                        print("loss:", loss)
                        print("KLD:", kld_loss)
                        print("BCE:", bce_loss)
                        assert False, "Grad explode"
                # for name, param in self.model.vae.named_parameters():
                #     print(name, param)
                #     print("grad:", param.grad)
                # exit()
                instance_loss = loss.item() / len(batch)
                tqdm_train.set_description(f"[Epoch {epoch:02d}] loss: {instance_loss:7.4f}", refresh=False)
                if step % 1000 == 0:
                    sw.add_scalar("Batch_vae_loss", instance_loss, global_step=step)
                opt_vae.step()
                if is_debug:
                    for name, param in self.model.vae.named_parameters():
                        if bool(torch.isnan(param).any()):
                            print(name, param)
                            # print("grad:", param.grad)
                            assert False, "Value explode"

            # ---- evaluation
            if self.is_swa:  # To evaluate model
                opt_vae.swap_swa_sgd()

            if is_debug:
                train_loss = self.eval_vae(dl_train)
            else:
                train_loss = -1
            valid_loss = self.eval_vae(dl_valid)
            print(f"[Epoch {epoch:02d}] epoch_train_loss: {train_loss:0.5f}, epoch_valid_loss: {valid_loss:0.5f}")
            sw.add_scalars("Epoch_vae_loss", {"train": train_loss, "valid": valid_loss}, global_step=epoch)
            sw.flush()
            if curr_min_loss > valid_loss:
                curr_min_loss = valid_loss
                torch.save(self.model.vae.state_dict(), self.vae_path_curr)
                curr_patience = 0
                self.best_epoch_vae = epoch
            else:
                curr_patience += 1
            if epoch <= min_epoch:
                curr_patience = 0
            elif float(valid_loss) > 10000 or np.isnan(float(valid_loss)):
                curr_patience = max_patience

            self.last_epoch_vae = epoch
            if curr_patience == max_patience:
                break

            if self.is_swa:  # To continue learning model
                opt_vae.swap_swa_sgd()

        # ----- done vae training ----
        os.replace(self.vae_path_curr, self.vae_path)
        self.model.vae.load_state_dict(torch.load(self.vae_path))  # updated from best model
        torch.save(self.model.vae.state_dict(), self.vae_path)

    def _train_cardnet(self, train_data, valid_data=None, test_data=None):
        self.model.to(self.device)
        dl_train = self._dataloader(train_data, shuffle=True)
        if valid_data:
            dl_valid = self._dataloader(valid_data, shuffle=False)
        if test_data:
            dl_test = self._dataloader(test_data, shuffle=False)

        # ---- Estimator training
        opt_model = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2)
        if self.is_swa:
            opt_model = SWA(opt_model, swa_start=5, swa_freq=2, swa_lr=self.lr / 2)

        min_epoch = 10
        max_patience = self.patience
        curr_patience = 0
        curr_min_q_err = 10000
        self.model_path_curr = self.save_dir + "/_best_model.pt"
        sw = SummaryWriter(self.logdir)
        step = 0
        print("train_cardnet")
        for epoch in range(1, self.max_epoch + 1):
            tqdm_train = tqdm(dl_train, mininterval=5)
            # ---- train loop
            for batch in tqdm_train:
                opt_model.zero_grad()
                step += 1
                x, c = self._batch_to_device(batch)
                loss = self.model.loss(x, c)
                loss.backward()
                if self.clip_gr > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_gr)
                c_hat = self.model(x)
                q_error = ut.q_error(c.cpu().numpy().reshape(-1), c_hat.detach().cpu().numpy().reshape(-1))
                tqdm_train.set_description(f"[Epoch {epoch:02d}] loss: {loss:7.4f} q_err: {q_error:7.4f}",
                                           refresh=False)
                if step % 1000 == 0:
                    sw.add_scalars("Batch", {"loss": loss.item(), "q_err": q_error}, global_step=step)
                opt_model.step()

            # ---- evaluation
            if self.is_swa:  # To evaluate model
                opt_model.swap_swa_sgd()

            if is_debug:
                train_loss, train_q_error = self.eval_cardnet(dl_train)
            else:
                train_loss, train_q_error = -1, -1
            valid_loss, valid_q_error = self.eval_cardnet(dl_valid, update_dynamic_weight=True)
            print(
                f"[Epoch {epoch:02d}] train [loss, q_err]: [{train_loss:.6f}, {train_q_error:.4f}], valid [loss, q_err]: [{valid_loss:.6f}, {valid_q_error:.4f}]")
            sw.add_scalars("Epoch_loss", {"train": train_loss, "valid": valid_loss}, global_step=epoch)
            sw.add_scalars("Epoch_q_err", {"train": train_q_error, "valid": valid_q_error}, global_step=epoch)
            sw.flush()
            if curr_min_q_err > valid_q_error:
                curr_min_q_err = valid_q_error
                torch.save(self.model.state_dict(), self.model_path_curr)
                curr_patience = 0
                self.best_epoch = epoch
            else:
                curr_patience += 1

            if epoch <= min_epoch:
                curr_patience = 0

            self.last_epoch = epoch
            if curr_patience == max_patience:
                break

            if self.is_swa:  # To continue learning model
                opt_model.swap_swa_sgd()

        # ----- done vae training ----
        os.replace(self.model_path_curr, self.save_path)

        self.model.load_state_dict(torch.load(self.save_path))  # updated from best model
        torch.save(self.model.state_dict(), self.save_path)
