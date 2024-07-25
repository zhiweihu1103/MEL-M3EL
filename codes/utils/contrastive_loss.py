import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.contiguous()

def normalize(t, dim, eps=1e-6):
    return F.normalize(t, dim=dim, eps=eps)

def gather_cat(x: torch.Tensor, grad=False, contiguous_grad=False) -> torch.Tensor:
    if not grad:
        gathers = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(gathers, x)
    else:
        gathers = torch.distributed.nn.all_gather(x)

    if x.ndim == 0:
        gathers = torch.stack(gathers)
    else:
        gathers = torch.cat(gathers)

    if contiguous_grad:
        gathers = ContiguousGrad.apply(gathers)

    return gathers

class InfoNCELoss(nn.Module):
    def __init__(self, T_init=0.07, **kwargs):
        super().__init__()
        self.tau = T_init
        self.xe = nn.CrossEntropyLoss(reduction='none')

    def forward(self, image_emb, text_emb):
        n = image_emb.shape[0]
        logits = image_emb @ text_emb.T
        labels = torch.arange(n).cuda()
        loss_t = self.xe(logits, labels)
        loss_i = self.xe(logits.T, labels)
        loss = (loss_i + loss_t) / 2
        loss = loss.mean()
        return loss

class MCLETLoss(nn.Module):
    def __init__(self, embedding_dim, cl_temperature=0.05):
        super(MCLETLoss, self).__init__()
        self.cl_temperature = cl_temperature
        self.embedding_dim = embedding_dim
        self.cl_fc = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=True),
            nn.ELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=True),
        )

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def forward(self, A_embedding, B_embedding):
        tau = self.cl_temperature
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.cl_fc(A_embedding)
        B_embedding = self.cl_fc(B_embedding)

        refl_sim_1 = f(self.sim(A_embedding, A_embedding))
        between_sim_1 = f(self.sim(A_embedding, B_embedding))
        loss_1 = -torch.log(between_sim_1.diag() / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))

        refl_sim_2 = f(self.sim(B_embedding, B_embedding))
        between_sim_2 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(between_sim_2.diag() / (refl_sim_2.sum(1) + between_sim_2.sum(1) - refl_sim_2.diag()))

        loss = (loss_1 + loss_2) * 0.5
        loss = loss.mean()

        return loss

class WeightedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.03, inter_weight=1.0, intra_weight=0.8, logger=None):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature
        self.logger = logger
        self.inter_weight = inter_weight
        self.intra_weight = intra_weight

    def compute_loss(self, logits, mask):
        return - torch.log((F.softmax(logits, dim=1) * mask).sum(1))

    def _get_positive_mask(self, batch_size):
        diag = np.eye(batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask)
        return mask.cuda(non_blocking=True)

    def forward(self, entity_features, mention_features):
        batch_size = entity_features.shape[0]

        # Normalize features
        entity_features = nn.functional.normalize(entity_features, dim=1)
        mention_features = nn.functional.normalize(mention_features, dim=1)

        # Inter-modality alignment
        inter_entity = entity_features @ mention_features.t()
        inter_mention = mention_features @ entity_features.t()

        # Intra-modality alignment
        inner_entity = entity_features @ entity_features.t()
        inner_mention = mention_features @ mention_features.t()

        inter_entity /= self.temperature
        inter_mention /= self.temperature
        inner_entity /= self.temperature
        inner_mention /= self.temperature

        positive_mask = self._get_positive_mask(entity_features.shape[0])
        inner_entity = inner_entity * positive_mask
        inner_mention = inner_mention * positive_mask

        entity_logits = torch.cat([self.inter_weight * inter_entity, self.intra_weight * inner_entity], dim=1)
        mention_logits = torch.cat([self.inter_weight * inter_mention, self.intra_weight * inner_mention], dim=1)

        diag = np.eye(batch_size)
        mask_entity = torch.from_numpy((diag)).cuda()
        mask_mention = torch.from_numpy((diag)).cuda()

        mask_neg_entity = torch.zeros_like(inner_entity)
        mask_neg_mention = torch.zeros_like(inner_mention)
        mask_entity = torch.cat([mask_entity, mask_neg_entity], dim=1)
        mask_mention = torch.cat([mask_mention, mask_neg_mention], dim=1)

        loss_entity = self.compute_loss(entity_logits, mask_entity)
        loss_mention = self.compute_loss(mention_logits, mask_mention)

        return ((loss_entity.mean() + loss_mention.mean())) / 2