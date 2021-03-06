import torch
import torch.nn as nn
from torch.nn import functional as tnf

class GlimpseNetwork(nn.Module):
    """Glimpse netowrk

    Glimpse network described in the paper
    http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

    Args:
        input_size: The number of expected features in the input x
        location_size: The number of expected features in the input location
        internal_size: The number of expected features in h_g and h_l described
            in the paper
        output_size: The number of expected features in the output

    Inputs: x, location
        - x of shape (batch_size, input_size): tensor containing features of 
          the input batched images
        - location of shape (batch_size, location_size): tensor containing
          features of input batched locations

    Outputs: output
        - output of shape (batch_size, output_size): tensor containing features
          of glimpse g described in the paper
    """
    def __init__(self, input_size, location_size, internal_size, output_size):
        super(GlimpseNetwork, self).__init__()
        self.fc_g = nn.Linear(input_size, internal_size)
        self.fc_l = nn.Linear(location_size, internal_size)
        self.fc_gg = nn.Linear(internal_size, output_size)
        self.fc_lg = nn.Linear(internal_size, output_size)

    def forward(self, glimpse, location):
        hg = tnf.relu(self.fc_g(glimpse))
        hl = tnf.relu(self.fc_l(location))
        output = tnf.relu(self.fc_gg(hg) + self.fc_lg(hl))
        return output

class CoreNetwork(nn.Module):
    """Core network

    Core network described in the paper
    http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of expected features in hidden states of RNN

    Inputs: g, prev_h
        - g of shape (batch_size, input_size): tensor containing glimpse features
          computed by GlimpseNetwork
        - prev_h of shape (batch_size, hidden_size): tensor containing the previous
          hidden state

    Outputs: h
        - h of shape (batch_size, hidden_size): tensor containing the current
          hidden state
    """
    def __init__(self, input_size, hidden_size):
        super(CoreNetwork, self).__init__()
        self.hidden_size    = hidden_size
        #self.rnn_cell      = nn.RNNCell(input_size, hidden_size, nonlinearity='relu')
        self.rnn_cell       = nn.LSTMCell(input_size, hidden_size)

    def forward(self, g, prev_h, prev_c):
        #h = self.rnn_cell(g, prev_h)
        ht, ct = self.rnn_cell(g, (prev_h, prev_c))
        return ht, ct

    '''def init_hidden(self, batch_size):
        hidden_layer = torch.zeros(batch_size, self.hidden_size).to(self.device)
        return hidden_layer'''

class LocationNetwork(nn.Module):
    """Location network

    Location network described in the paper
    http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

    Args:
        input_size: The number of expected features in the input x
        output_size: The number of expected features in the output

    Inputs: x, location
        - x of shape (batch_size, input_size): tensor containing the output of
          CoreNetwork

    Outputs: lt, loc_p
        - output of shape (batch_size, output_size): tensor containing features
          of the next location l described in the paper
        - loc_p of shape (batch_size, 1): tensor containing the log probability
          of the location
    """
    '''Paper original: The location network outputs the mean of the location policy at timetand is defined as 
       fl(h) = Linear(h) where his the state of the core network/RNN 
       New way: fl(fc(h))'''
    def __init__(self, input_size, location_size, std):
        super(LocationNetwork, self).__init__()

        self.std    = std
        #self.fc    = nn.Linear(input_size, location_size)
        hidden_size = input_size // 2
        self.fc     = nn.Linear(input_size, hidden_size)
        self.fl     = nn.Linear(hidden_size, location_size)

    def forward(self, ht):
        #mu = torch.tanh(self.fc(ht.detach()))
        feature = tnf.relu(self.fc(ht.detach()))
        mu      = torch.tanh(self.fl(feature))

        '''if self.training:
            distribution    = torch.distributions.Normal(mu, self.std)
            lt              = distribution.sample()
            loc_p           = distribution.log_prob(lt)
            loc_p           = torch.sum(loc_p, dim=1)
            lt              = torch.clamp(lt,-1.0, 1.0)
        else:
            output  = mu
            loc_p   = torch.ones(output.size(0))'''

        distribution    = torch.distributions.Normal(mu, self.std)
        lt              = distribution.sample()
        loc_p           = distribution.log_prob(lt)
        loc_p           = torch.sum(loc_p, dim=1)
        lt              = torch.clamp(lt, -1.0, 1.0)

        return lt, loc_p

#in ppo its a actor network
class ActorNetwork(nn.Module):
    """Action network

    Action network described in the paper
    http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

    Args:
        input_size: The number of expected features in the input x
        output_size: The number of expected features in the output

    Inputs: x
        - x of shape (batch_size, input_size): tensor containing the output of
          CoreNetwork

    Outputs: logit
        - logit of shape (batch_size, output_size): tensor containing the logit
          of the predicted actions.
    """
    def __init__(self, input_size, output_size):
        super(ActorNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, ht):
        '''logit = tnf.log_softmax(self.fc(ht), dim=1) #CrossEntropyLoss combines log_softmax and NLLLoss'''
        act_probs = self.fc(ht)
        return act_probs

#in ppo its a critic network
class CriticNetwork(nn.Module):
    """Baseline network

    Baseline network described in the paper
    http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
    Here I use the simple version which only contains a simple
    fully connected neural network with sigmoid function

    Args:
        input_size: The number of expected features in the input x
        output_size: The number of expected features in the output

    Inputs: x
        - x of shape (batch_size, input_size): tensor containing the output of
          CoreNetwork

    Outputs: output
        - output of shape (batch_size, output_size): tensor containing the
          predicted rewards
    """
    def __init__(self, input_size, output_size):
        super(CriticNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, ht):
        output  = torch.sigmoid(self.fc(ht.detach()))
        return output
