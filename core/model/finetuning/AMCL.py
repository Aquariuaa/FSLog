import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn import Parameter
from core.utils import accuracy, PRF
from .finetuning_model import FinetuningModel

class NegLayer(nn.Module):
    # self.feat_dim, self.test_way, self.margin, self.scale_factor
    def __init__(self, in_features, out_features, margin=0.40, scale_factor=30.0):
        super(NegLayer, self).__init__()
        self.margin = margin
        self.scale_factor = scale_factor
        # generate weights
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        # F.normalize l2 xi *wT
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        if label is None:
            return cosine * self.scale_factor
        phi = cosine - self.margin
        # cosine shape is test_way, label is [0, 1, 0, 1]
        output = torch.where(self.one_hot(label, cosine.shape[1]).byte(), phi, cosine)
        output *= self.scale_factor
        return output

    def one_hot(self, y, num_class):
        return (
            torch.zeros((len(y), num_class)).to(y.device).scatter_(1, y.unsqueeze(1), 1)
        )


class DistLinear(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DistLinear, self).__init__()
        # in_channel=9600 out_channel=2
        self.fc = nn.Linear(in_channel, out_channel, bias=False)
        self.class_wise_learnable_norm = True
        # normalize the weights
        if self.class_wise_learnable_norm:
            weight_norm(self.fc, "weight", dim=0)
        # reference large-margin softmax loss
        self.scale_factor = 2

    def forward(self, x):
        # L2, dim1
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        cos_dist = self.fc(x_normalized)
        score = self.scale_factor * cos_dist
        return score

class AMCL(FinetuningModel):
    def __init__(self, feat_dim, num_class, inner_param, **kwargs):
        super(AMCL, self).__init__(**kwargs)
        # the tcn kernel is 32, the feat_dim is 9600
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.margin = -0.05
        self.scale_factor = 2
        self.inner_param = inner_param
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = DistLinear(self.feat_dim, self.num_class)

    def set_forward(self, batch):
        logs, global_target = batch
        logs = logs.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(logs.view(-1, 300, 300))
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_negative_margin(
                support_feat[i], support_target[i], query_feat[i]
            )
            output_list.append(output)
        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.reshape(-1))
        res = PRF(output, query_target.reshape(-1))
        # return output, acc, res
        return output, acc

    def set_forward_loss(self, batch):
        logs, target = batch
        logs = logs.to(self.device)
        target = target.to(self.device)
        feat = self.emb_func(logs.view(-1,300,300))
        output = self.classifier(feat)
        loss = self.loss_func(output, target.reshape(-1))
        acc = accuracy(output, target.reshape(-1))
        return output, acc, loss

    def set_forward_negative_margin(self, support_feat, support_target, query_feat):
        classifier = NegLayer(self.feat_dim, self.test_way, self.margin, self.scale_factor)
        optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])
        classifier = classifier.to(self.device)
        classifier.train()
        loss_func = nn.CrossEntropyLoss()
        support_size = support_feat.size(0)
        batch_size = 4
        for epoch in range(self.inner_param["inner_train_iter"]):
            rand_id = torch.randperm(support_size)
            for i in range(0, support_size, batch_size):
                # second train before test
                select_id = rand_id[i : min(i + batch_size, support_size)]
                batch = support_feat[select_id]
                target = support_target[select_id]

                output = classifier(batch, target)
                loss = loss_func(output, target)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)  # retain_graph = True
                optimizer.step()
        output = classifier(query_feat)
        return output