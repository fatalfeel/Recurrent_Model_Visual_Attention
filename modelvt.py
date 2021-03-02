import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.nn import functional as tnf
from modules import GlimpseNetwork, LocationNetwork, CoreNetwork, ActorNetwork, CriticNetwork

class ModelVT(nn.Module):
    """Reccurrent Attention Model

    Reccurrent Attention Model described in the paper
    http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

    Args:
        location_size: The number of expected features in location, 2 for MNIST dataset
        location_std: The standard deviation used by location network
        num_classes: The number of expected features in action, 10 for MNIST dataset
        glimpse_size: The number of expected size of initial glimpse, 8 for MNIST dataset
            which gives a cropped image of 8X8 centered on the given location 
        num_glimpses: The number of expected glimpses
        num_scales: The number of scale of a glimpse
        feature_size: The number of expected features in h_g and h_l described
            in the paper
        glimpse_feature_size: The number of expected features in glimpse feature,
            which is g described in the paper
        hidden_size: The number of expected features in the hidden state of RNN

    Inputs: x
        - x of shape (batch_size, channel, heigth, width): tensor containing features
          of the input images, (batch_size, 1, 28, 28) for MNIST dataset

    Outputs: action_log_probs, locations, location_log_probs, baselines
        - action_log_probs of shape (batch_size, num_classes): tensor containing the
          log probabilities of the predicted actions
        - locations of shape (batch_size, glimpse_size): tensor containing the locations
          for each glimpse
        - location_log_probs of shape (batch_size, location_size): tensor containing the
          log probabilities of the predicted locations
        - baselines of shape (batch_size, glimpse_size): tensor containing the
          the predicted rewards
    """

    def __init__(self,
                 location_size,
                 num_classes,
                 location_std,
                 glimpse_size,
                 num_glimpses,
                 num_scales,
                 feature_size,
                 glimpse_feature_size,
                 hidden_size,
                 model_device):
        super(ModelVT, self).__init__()

        self.location_size  = location_size
        self.glimpse_size   = glimpse_size
        self.num_glimpses   = num_glimpses
        self.num_scales     = num_scales
        # compute input size after retina encoding
        self.input_size     = glimpse_size * glimpse_size * num_scales

        self.glimpse_network    = GlimpseNetwork(self.input_size, location_size, feature_size, glimpse_feature_size)
        self.core_network       = CoreNetwork(glimpse_feature_size, hidden_size)
        self.fl                 = LocationNetwork(hidden_size, location_size, location_std)
        self.fa                 = ActorNetwork(hidden_size, num_classes)
        self.critic_network     = CriticNetwork(hidden_size, 1) #CriticNetwork = BaselineNetwork
        self.device             = model_device

    '''def init_location(self, batch_size):
        return torch.zeros(batch_size, self.location_size)'''

    def show_MNIST(self, img):
        grid = torchvision.utils.make_grid(img)
        trimg = grid.numpy().transpose(1, 2, 0)
        '''cv2.imshow('MNIST', trimg)
        cv2.waitKey(2000);
        cv2.destroyAllWindows()'''
        plt.imshow(trimg)
        plt.title('Batch from dataloader')
        plt.axis('off')
        plt.show()

    #practice (i.) affine grid & grid sample source - affinegrid_gridsample.zip
    def Retina(self, data, location, output_size, nsc):
        #Transform image to retina representation
        batch_size, input_size = data.size(0), data.size(2)-1

        scale   = output_size / input_size
        orisize = torch.Size([batch_size, 1, output_size, output_size])

        # construct theta for affine transformation
        theta           = torch.zeros(batch_size, 2, 3).to(self.device)
        theta[:, :, 2]  = location
        output          = torch.zeros(batch_size, output_size * output_size * nsc).to(self.device)

        for i in range(nsc):
            # theta is location shift
            theta[:, 0, 0] = scale
            theta[:, 1, 1] = scale

            grid = tnf.affine_grid(theta, orisize, align_corners=True)

            #glimpse = tnf.grid_sample(data, grid).view(batch_size, -1)
            sample = tnf.grid_sample(data, grid, align_corners=True)
            #self.show_MNIST(sample)
            glimpse = sample.view(batch_size, -1)

            #output[:, i * output_size*output_size: (i + 1) * output_size*output_size] = glimpse
            fillsize    = output_size*output_size
            pos_start   =  i    * fillsize
            pos_end     = (i+1) * fillsize
            output[:, pos_start : pos_end] = glimpse
            scale *= 2.0

        return output.detach()

    def forward(self, data):
        batch_size  = data.size(0)
        #ht          = self.core_network.init_hidden(batch_size)
        ht          = torch.zeros(batch_size, self.core_network.hidden_size).to(self.device)
        ct          = torch.zeros(batch_size, self.core_network.hidden_size).to(self.device)

        #location = self.init_location(batch_size)
        location            = torch.FloatTensor(batch_size, self.location_size).uniform_(-1, 1).to(self.device)
        location_log_probs  = torch.zeros(batch_size,       self.num_glimpses).to(self.device)
        locations           = torch.zeros(batch_size,       self.num_glimpses, self.location_size).to(self.device)
        critic_values       = torch.zeros(batch_size,       self.num_glimpses).to(self.device)

        for i in range(self.num_glimpses):
            #paper p4. low-resolution representation as a glimpse
            glimpse                     = self.Retina(data, location.detach(), self.glimpse_size, self.num_scales)
            gt                          = self.glimpse_network(glimpse, location)
            ht, ct                      = self.core_network(gt, ht, ct)
            location, log_prob          = self.fl(ht)
            locations[:, i]             = location
            location_log_probs[:, i]    = log_prob
            cvalue                      = self.critic_network(ht)
            critic_values[:, i]         = cvalue.squeeze()

        act_probs = self.fa(ht) #classifier

        return act_probs, locations, location_log_probs, critic_values
