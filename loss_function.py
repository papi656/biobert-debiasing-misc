import torch 
from torch import nn
from torch.nn import functional as F
import numpy as np

class ClfDebiasLossFunction(nn.Module):
    def forward(self, num_labels, hidden, logits, bias, teacher_probs, labels):
        """
        :param hidden: [batch, n_features] hidden features from the model
        :param logits: [batch, n_classes] logit score for each class
        :param bias: [batch, n_classes] log-probabilties from the bias for each class
        :param labels: [batch] integer class labels
        :return: scalar loss
        """
        raise NotImplementedError()

class Plain(ClfDebiasLossFunction):
    def forward(self, num_labels, hidden, logits, bias, teacher_probs, labels):
        return F.cross_entropy(logits.view(-1, num_labels), labels.view(-1))

class BiasProduct(ClfDebiasLossFunction):
    def forward(self, num_labels, hidden, logits, bias, teacher_probs, labels):
        lg_softmax_op = nn.LogSoftmax(dim=2)
        # print(logits.shape)
        # print(bias.shape)
        # applying log softmax to main models logits
        logits = logits.float() # in case we were in fp16 mode
        logits_log = lg_softmax_op(logits)
        # applying log softmax to bias models logits
        bias = bias.float() # to make sure dtype=float32 
        bias_log = torch.log(bias)

        return F.cross_entropy((logits_log + bias_log).view(-1, num_labels), labels.view(-1))

class LearnedMixinH(ClfDebiasLossFunction):
    def __init__(self, penalty):
        super().__init__()
        self.penalty = penalty
        self.bias_lin = torch.nn.Linear(768,1)

    def forward(self, num_labels, hidden, logits, bias, teacher_probs, labels):
        logits = logits.float()
        lg_softmax_op = nn.LogSoftmax(dim=2)
        logits_log = lg_softmax_op(logits)
        factor = self.bias_lin.forward(hidden)
        factor = factor.float()
        factor = F.softplus(factor)

        bias_log = torch.log(bias)
        bias_log = bias_log * factor

        # bias_lp = F.softmax(bias_log, dim=2) -- used earlier, got -ve loss, replaced with log-softmax
        bias_lp = lg_softmax_op(bias_log)
        entropy = -(torch.exp(bias_lp) * bias_lp).sum(2).mean(1)
        # entropy = -(torch.exp(bias_lp) * bias_lp).sum(2).mean(1)

        loss = F.cross_entropy((logits_log + bias_log).view(-1, num_labels), labels.view(-1)) + self.penalty * entropy 
        
        loss = loss.sum()
        return loss


class Reweight(ClfDebiasLossFunction):
    def forward(self, num_labels, hidden, logits, bias, teacher_probs, labels):
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, num_labels), labels.view(-1), reduction='none')
        one_hot_labels = torch.eye(logits.size(2)).cuda()[labels]
        weights = 1 - (one_hot_labels * bias).sum(2)
        return (weights.view(-1) * loss).sum() / weights.sum()

class DistillLoss(ClfDebiasLossFunction):
    def forward(self, num_labels, hidden, logits, bias, teacher_probs, labels):
        softmax_op = torch.nn.Softmax(dim=2)
        probs = softmax_op(logits)

        example_loss = -(teacher_probs * probs.log()).sum(2)
        batch_loss = example_loss.mean()

        return batch_loss 

class SmoothedDistillLoss(ClfDebiasLossFunction):
    def forward(self, num_labels, hidden, logits, bias, teacher_probs, labels):
        softmax_op = torch.nn.Softmax(dim=2)
        probs = softmax_op(logits) # probs from student model

        one_hot_labels = torch.eye(logits.size(2)).cuda()[labels]
        # the torch.exp is just for numerical stabilization
        # maybe try running once without torch.exp()
        weights = (1 - (one_hot_labels * bias).sum(2))
        # weights = (1 - (one_hot_labels * torch.exp(bias)).sum(2))
        weights = weights.unsqueeze(2).expand_as(teacher_probs)

        exp_teacher_probs = teacher_probs ** weights 
        norm_teacher_probs = exp_teacher_probs / exp_teacher_probs.sum(2).unsqueeze(2).expand_as(teacher_probs)

        example_loss = -(norm_teacher_probs * probs.log()).sum(2)
        batch_loss = example_loss.mean()

        return batch_loss

