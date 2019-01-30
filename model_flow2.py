import nn
from nn import lecun_uniform, LearnedUpsampling1d, concat_init
import utils

import torch
from torch.nn import functional as F
from torch.nn import init

import numpy as np


class SampleRNN(torch.nn.Module):

    def __init__(self, frame_sizes, n_rnn, dim, learn_h0, q_levels,
                 weight_norm):
        super().__init__()

        self.dim = dim
        self.q_levels = q_levels

        ns_frame_samples = map(int, np.cumprod(frame_sizes))
        self.frame_level_rnns = torch.nn.ModuleList([
            FrameLevelRNN(
                frame_size, n_frame_samples, n_rnn, dim, learn_h0, weight_norm
            )
            for (frame_size, n_frame_samples) in zip(
                frame_sizes, ns_frame_samples
            )
        ])

        self.sample_level_mlp = SampleLevelMLP(
            frame_sizes[0], dim, q_levels, weight_norm
        )

    @property
    def lookback(self):
        return self.frame_level_rnns[-1].n_frame_samples


class FrameLevelRNN(torch.nn.Module):

    def __init__(self, frame_size, n_frame_samples, n_rnn, dim,
                 learn_h0, weight_norm):
        super().__init__()

        self.frame_size = frame_size
        self.n_frame_samples = n_frame_samples
        self.dim = dim

        h0 = torch.zeros(n_rnn, dim)
        if learn_h0:
            self.h0 = torch.nn.Parameter(h0)
        else:
            self.register_buffer('h0', torch.autograd.Variable(h0))

        self.input_expand = torch.nn.Conv1d(
            in_channels=n_frame_samples,
            out_channels=dim,
            kernel_size=1
        )
        init.kaiming_uniform(self.input_expand.weight)
        init.constant(self.input_expand.bias, 0)
        if weight_norm:
            self.input_expand = torch.nn.utils.weight_norm(self.input_expand)

        self.rnn = torch.nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=n_rnn,
            batch_first=True
        )
        for i in range(n_rnn):
            concat_init(
                getattr(self.rnn, 'weight_ih_l{}'.format(i)),
                [lecun_uniform, lecun_uniform, lecun_uniform]
            )
            init.constant(getattr(self.rnn, 'bias_ih_l{}'.format(i)), 0)

            concat_init(
                getattr(self.rnn, 'weight_hh_l{}'.format(i)),
                [lecun_uniform, lecun_uniform, init.orthogonal]
            )
            init.constant(getattr(self.rnn, 'bias_hh_l{}'.format(i)), 0)

        self.upsampling = LearnedUpsampling1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=frame_size
        )
        init.uniform(
            self.upsampling.conv_t.weight, -np.sqrt(6 / dim), np.sqrt(6 / dim)
        )
        init.constant(self.upsampling.bias, 0)
        if weight_norm:
            self.upsampling.conv_t = torch.nn.utils.weight_norm(
                self.upsampling.conv_t
            )

    def forward(self, prev_samples, upper_tier_conditioning, hidden):
        (batch_size, _, _) = prev_samples.size()

        input = self.input_expand(
          prev_samples.permute(0, 2, 1)
        ).permute(0, 2, 1)
        if upper_tier_conditioning is not None:
            input += upper_tier_conditioning

        reset = hidden is None

        if hidden is None:
            (n_rnn, _) = self.h0.size()
            hidden = self.h0.unsqueeze(1) \
                            .expand(n_rnn, batch_size, self.dim) \
                            .contiguous()

        (output, hidden) = self.rnn(input, hidden)

        output = self.upsampling(
            output.permute(0, 2, 1)
        ).permute(0, 2, 1)
        return (output, hidden)


class SampleLevelMLP(torch.nn.Module):

    def __init__(self, frame_size, dim, q_levels, weight_norm):
        super().__init__()

        self.q_levels = q_levels

        self.input = torch.nn.Conv1d(
            in_channels=frame_size,
            out_channels=dim,
            kernel_size=1,
            bias=False
        )
        init.kaiming_uniform(self.input.weight)
        if weight_norm:
            self.input = torch.nn.utils.weight_norm(self.input)

        self.hidden = torch.nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1
        )
        init.kaiming_uniform(self.hidden.weight)
        init.constant(self.hidden.bias, 0)
        if weight_norm:
            self.hidden = torch.nn.utils.weight_norm(self.hidden)

        self.output = torch.nn.Conv1d(
            in_channels=dim,
            out_channels=2, # mu, log_sigma
            kernel_size=1
        )
        lecun_uniform(self.output.weight)
        init.constant(self.output.bias, 0)
        if weight_norm:
            self.output = torch.nn.utils.weight_norm(self.output)

    def forward(self, prev_samples, upper_tier_conditioning):
        upper_tier_conditioning = upper_tier_conditioning.permute(0, 2, 1)

        x = F.relu(self.input(prev_samples) + upper_tier_conditioning)
        x = F.relu(self.hidden(x))
        x = self.output(x).permute(0, 2, 1).contiguous()

        clamped_x = torch.cat( (x[:, :, 0].unsqueeze(dim=2), 
                                x[:, :, 1].clamp(min=-8).unsqueeze(dim=2) )
                                , dim=2)
        return clamped_x    # mu = x[:, :, 0], log_sigma = clamped_x[:, :, 1]


class Runner:

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.reset_hidden_states()

    def reset_hidden_states(self):
        self.hidden_states = {rnn: None for rnn in self.model.frame_level_rnns}

    def run_rnn(self, rnn, prev_samples, upper_tier_conditioning):
        (output, new_hidden) = rnn(
            prev_samples, upper_tier_conditioning, self.hidden_states[rnn]
        )
        self.hidden_states[rnn] = new_hidden.detach()
        return output


class Predictor(Runner, torch.nn.Module):

    def __init__(self, model):
        super().__init__(model)

    def forward(self, input_sequences, reset):
        if reset:
            self.reset_hidden_states()

        (batch_size, seq_len) = input_sequences.size()
        seq_len = seq_len - self.model.lookback # seq_len = 1024

        upper_tier_conditioning = None
        for rnn in reversed(self.model.frame_level_rnns):
            from_index = self.model.lookback - rnn.n_frame_samples
            to_index = -rnn.n_frame_samples # + 1
            prev_samples = input_sequences[:, from_index : to_index]
            prev_samples = prev_samples.contiguous().view(
                batch_size, -1, rnn.n_frame_samples
            )

            upper_tier_conditioning = self.run_rnn(
                rnn, prev_samples, upper_tier_conditioning
            )

        bottom_frame_size = self.model.frame_level_rnns[0].frame_size

        # break seq_len -> (seq_len/bottom_frame_size, bottom_frame_size)
        # note that (bottom_frame_size, seq_len/bottom_frame_size) will not work
        mlp_input_sequences = input_sequences[:, self.model.lookback:].view(
            batch_size, (seq_len // bottom_frame_size), bottom_frame_size
        )
        upper_tier_conditioning = upper_tier_conditioning.view(
            batch_size, (seq_len // bottom_frame_size), bottom_frame_size, self.model.dim
        )

        # second last dimension to be kept same as in_channel of first layer of MLP
        # [**] may want to replace zeros() with randn() here
        z = mlp_input_sequences.new_zeros(batch_size, bottom_frame_size, seq_len // bottom_frame_size,
                requires_grad=False)
        # if(self.cuda):
        #     z = z.cuda()
        
        sample_dist = []

        for j in range(bottom_frame_size):

            flow_out = self.model.sample_level_mlp(
                z, upper_tier_conditioning[:, :, j, :])
            
            sample_dist.append(flow_out)

            new_z = (mlp_input_sequences[:, :, j] - flow_out[:, :, 0]) \
                            / torch.exp(flow_out[:, :, 1])
            z = torch.cat([z[:, 1:, :], new_z.unsqueeze(1)], dim=1)
            z = z.clamp(min=-5, max=5)
        # stack along dim=2 , so that outputs are not concatened rather inter-leaved
        # dimesion changes: (batch, seq_len/fs, 2) -> (batch, seq_len/fs, fs, 2) -> (btch, seq_len, 2)
        return torch.stack(sample_dist, dim=2).view(batch_size, seq_len, 2)


class Generator(Runner):

    def __init__(self, model, cuda=False):
        super().__init__(model)
        self.cuda = cuda


    def get_rand_z2(self, n_seqs, seq_len, bottom_frame_size):
        ''' calculating sequence of z as required by IAF during sampling
        '''
        rand_z = torch.randn(n_seqs, seq_len)
        idx = torch.arange(0, seq_len, bottom_frame_size)

        # [**] may want to replace zeros() with randn() here
        z_temp = torch.zeros(n_seqs, bottom_frame_size, seq_len//bottom_frame_size)

        z_list = [z_temp]

        for j in range(0, bottom_frame_size-1):
            new_z = rand_z[:, idx+j]
            z_temp = torch.cat([ z_temp[:, 1:, :], new_z.unsqueeze(1) ], dim=1)
            z_list.append(z_temp)
        
        z = torch.stack(z_list, dim=3).view(n_seqs, bottom_frame_size, seq_len)
        return rand_z, z


    def get_rand_z(self, n_seqs, seq_len, bottom_frame_size):
        ''' calculating sequence of z as required by IAF during sampling
        '''
        rand_z = torch.randn(n_seqs, seq_len)

        # [**] may want to replace zeros() with randn() here
        z = torch.zeros(n_seqs, bottom_frame_size, seq_len)
        
        for i in range(1, bottom_frame_size+1):
            z[:, bottom_frame_size - i, i:] = rand_z[:, :-i]
        return rand_z, z


    def __call__(self, n_seqs, seq_len):
        # generation doesn't work with CUDNN for some reason
        torch.backends.cudnn.enabled = False

        self.reset_hidden_states()

        bottom_frame_size = self.model.frame_level_rnns[0].n_frame_samples
        sequences = torch.zeros(n_seqs, self.model.lookback + seq_len)
        frame_level_outputs = [None for _ in self.model.frame_level_rnns]

        rand_z, z = self.get_rand_z(n_seqs, seq_len, bottom_frame_size)

        if(self.cuda):
            z = z.cuda()
            rand_z = rand_z.cuda()
        
        for i in range(self.model.lookback, self.model.lookback + seq_len):
            for (tier_index, rnn) in \
                    reversed(list(enumerate(self.model.frame_level_rnns))):
                if i % rnn.n_frame_samples != 0:
                    continue

                prev_samples = torch.autograd.Variable(
                    sequences[:, i - rnn.n_frame_samples : i].unsqueeze(1),
                    volatile=True
                )
                if self.cuda:
                    prev_samples = prev_samples.cuda()

                if tier_index == len(self.model.frame_level_rnns) - 1:
                    upper_tier_conditioning = None
                else:
                    frame_index = (i // rnn.n_frame_samples) % \
                        self.model.frame_level_rnns[tier_index + 1].frame_size
                    upper_tier_conditioning = \
                        frame_level_outputs[tier_index + 1][:, frame_index, :] \
                                           .unsqueeze(1)

                frame_level_outputs[tier_index] = self.run_rnn(
                    rnn, prev_samples, upper_tier_conditioning
                )

            if  (i % bottom_frame_size != 0):
                continue

            upper_tier_conditioning = \
                frame_level_outputs[0]  # assuming that the frame-size will be same
                                        # as that of MLP level
            
            j = i - self.model.lookback # for indexing z, will use j insted of i

            sample_dist = self.model.sample_level_mlp(
                z[:, :, j:j+bottom_frame_size], upper_tier_conditioning
            ).data
            sequences[:, i : i+bottom_frame_size] = sample_dist[:, :, 0] \
                        + sample_dist[:, :, 1].exp() * rand_z[:, j:j+bottom_frame_size]

        torch.backends.cudnn.enabled = True

        return sequences[:, self.model.lookback :]
