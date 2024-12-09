import math
import os
from transformers import CLIPModel
from codes.utils.contrastive_loss import *
from codes.utils.utils import *

class M3ELEncoder(nn.Module):
    def __init__(self, args):
        super(M3ELEncoder, self).__init__()
        self.args = args
        current_directory = os.path.dirname(os.path.abspath(__file__))
        base_path = current_directory[0:current_directory.rfind('/')]
        self.base_path = base_path[0:base_path.rfind('/')]
        self.clip = CLIPModel.from_pretrained(self.base_path + self.args.pretrained_model)
        self.image_cls_fc = nn.Linear(self.args.model.input_hidden_dim, self.args.model.dv)
        self.image_tokens_fc = nn.Linear(self.args.model.input_image_hidden_dim, self.args.model.dv)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                pixel_values=None):
        clip_output = self.clip(input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values)

        text_embeds = clip_output.text_embeds
        image_embeds = clip_output.image_embeds

        text_seq_tokens = clip_output.text_model_output[0]
        image_patch_tokens = clip_output.vision_model_output[0]

        image_embeds = self.image_cls_fc(image_embeds)
        image_patch_tokens = self.image_tokens_fc(image_patch_tokens)
        return text_embeds, image_embeds, text_seq_tokens, image_patch_tokens

class TextIntraModalMatch(nn.Module):
    def __init__(self, args):
        super(TextIntraModalMatch, self).__init__()
        self.args = args
        self.fc_query = nn.Linear(self.args.model.input_hidden_dim, self.args.model.TIMM_hidden_dim)
        self.fc_key = nn.Linear(self.args.model.input_hidden_dim, self.args.model.TIMM_hidden_dim)
        self.fc_value = nn.Linear(self.args.model.input_hidden_dim, self.args.model.TIMM_hidden_dim)
        self.fc_cls = nn.Linear(self.args.model.input_hidden_dim, self.args.model.TIMM_hidden_dim)
        self.layer_norm = nn.LayerNorm(self.args.model.TIMM_hidden_dim)

    def forward(self,
                entity_text_cls,
                entity_text_tokens,
                mention_text_cls,
                mention_text_tokens):
        """

        :param entity_text_cls:     [num_entity, dim]
        :param entity_text_tokens:  [num_entity, max_seq_len, dim]
        :param mention_text_cls:    [batch_size, dim]
        :param mention_text_tokens: [batch_size, max_sqe_len, dim]
        :return:
        """
        entity_cls_fc = self.fc_cls(entity_text_cls)
        entity_cls_fc = entity_cls_fc.unsqueeze(dim=1)

        query = self.fc_query(entity_text_tokens)
        key = self.fc_key(mention_text_tokens)
        value = self.fc_value(mention_text_tokens)

        query = query.unsqueeze(dim=1)
        key = key.unsqueeze(dim=0)
        value = value.unsqueeze(dim=0)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.args.model.TIMM_hidden_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context = torch.matmul(attention_probs, value)
        context = torch.mean(context, dim=-2)
        context = self.layer_norm(context)

        g2l_matching_score = torch.sum(entity_cls_fc * context, dim=-1)
        g2l_matching_score = g2l_matching_score.transpose(0, 1)
        g2g_matching_score = torch.matmul(mention_text_cls, entity_text_cls.transpose(-1, -2))

        matching_score = (g2l_matching_score + g2g_matching_score) / 2

        return matching_score

class ImageIntraModalMatch(nn.Module):
    def __init__(self, args):
        super(ImageIntraModalMatch, self).__init__()
        self.args = args
        self.fc_query = nn.Linear(self.args.model.dv, self.args.model.IIMM_hidden_dim)
        self.fc_key = nn.Linear(self.args.model.dv, self.args.model.IIMM_hidden_dim)
        self.fc_value = nn.Linear(self.args.model.dv, self.args.model.IIMM_hidden_dim)
        self.fc_cls = nn.Linear(self.args.model.dv, self.args.model.IIMM_hidden_dim)
        self.layer_norm = nn.LayerNorm(self.args.model.IIMM_hidden_dim)

    def forward(self,
                entity_image_cls,
                entity_image_tokens,
                mention_image_cls,
                mention_image_tokens):
        """
        :param entity_image_cls:        [num_entity, dim]
        :param entity_image_tokens:     [num_entity, num_patch, dim]
        :param mention_image_cls:       [batch_size, dim]
        :param mention_image_tokens:    [batch_size, num_patch, dim]
        :return:
        """
        entity_cls_fc = self.fc_cls(entity_image_cls)
        entity_cls_fc = entity_cls_fc.unsqueeze(dim=1)

        query = self.fc_query(entity_image_tokens)
        key = self.fc_key(mention_image_tokens)
        value = self.fc_value(mention_image_tokens)

        query = query.unsqueeze(dim=1)
        key = key.unsqueeze(dim=0)
        value = value.unsqueeze(dim=0)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.args.model.IIMM_hidden_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context = torch.matmul(attention_probs, value)
        context = torch.mean(context, dim=-2)
        context = self.layer_norm(context)

        g2l_matching_score = torch.sum(entity_cls_fc * context, dim=-1)
        g2l_matching_score = g2l_matching_score.transpose(0, 1)
        g2g_matching_score = torch.matmul(mention_image_cls, entity_image_cls.transpose(-1, -2))

        matching_score = (g2l_matching_score + g2g_matching_score) / 2

        return matching_score

class CrossModalMatch(nn.Module):
    def __init__(self, args):
        super(CrossModalMatch, self).__init__()
        self.args = args
        self.text_fc = nn.Linear(self.args.model.input_hidden_dim, self.args.model.CMM_hidden_dim)
        self.image_fc = nn.Linear(self.args.model.dv, self.args.model.CMM_hidden_dim)
        self.gate_fc = nn.Linear(self.args.model.CMM_hidden_dim, 1)
        self.gate_act = nn.Tanh()
        self.gate_layer_norm = nn.LayerNorm(self.args.model.CMM_hidden_dim)

        self.match_module = MatchModule(self.args.model.CMM_hidden_dim)
        self.multi_head_module= MultiHeadModule(self.args.model.head_num, self.args.model.weight)
        self.fusion_module = FusionModule(self.args.model.CMM_hidden_dim)

        self.mclet_text_loss = MCLETLoss(embedding_dim=self.args.model.CMM_hidden_dim, cl_temperature=0.6)

    def forward(self, entity_text_cls, entity_image_tokens,
                mention_text_cls, mention_image_tokens):
        """
        :param entity_text_cls:         [num_entity, dim]
        :param entity_image_tokens:     [num_entity, num_patch, dim]
        :param mention_text_cls:        [batch_size, dim]
        :param mention_image_tokens:    [batch_size, num_patch, dim]
        :return:
        """
        entity_text_cls = self.text_fc(entity_text_cls)
        entity_text_cls_ori = entity_text_cls
        mention_text_cls = self.text_fc(mention_text_cls)
        mention_text_cls_ori = mention_text_cls

        entity_image_tokens = self.image_fc(entity_image_tokens)
        mention_image_tokens = self.image_fc(mention_image_tokens)

        entity_text_cls = self.match_module([entity_text_cls_ori.unsqueeze(dim=1), entity_image_tokens]).squeeze()
        entity_image_tokens = self.match_module([entity_image_tokens, entity_text_cls_ori.unsqueeze(dim=1)])
        entity_image_tokens = self.multi_head_module(entity_image_tokens)
        entity_context = self.fusion_module([entity_text_cls, entity_image_tokens])
        entity_gate_score = self.gate_act(self.gate_fc(entity_text_cls_ori))
        entity_context = self.gate_layer_norm((entity_text_cls_ori * entity_gate_score) + entity_context)

        mention_text_cls = self.match_module([mention_text_cls_ori.unsqueeze(dim=1), mention_image_tokens]).squeeze()
        mention_image_tokens = self.match_module([mention_image_tokens, mention_text_cls_ori.unsqueeze(dim=1)])
        mention_image_tokens = self.multi_head_module(mention_image_tokens)
        mention_context = self.fusion_module([mention_text_cls, mention_image_tokens])
        mention_gate_score = self.gate_act(self.gate_fc(mention_text_cls_ori))
        mention_context = self.gate_layer_norm((mention_text_cls_ori * mention_gate_score) + mention_context)

        score = torch.matmul(mention_context, entity_context.transpose(-1, -2))

        return score

class CrossModalMatchBidirection(nn.Module):
    def __init__(self, args):
        super(CrossModalMatchBidirection, self).__init__()
        self.args = args
        self.text_fc = nn.Linear(self.args.model.input_hidden_dim, self.args.model.CMM_hidden_dim)
        self.image_fc = nn.Linear(self.args.model.dv, self.args.model.CMM_hidden_dim)

        self.gate_fc = nn.Linear(self.args.model.CMM_hidden_dim, 1)
        self.gate_act = nn.Tanh()
        self.gate_layer_norm = nn.LayerNorm(self.args.model.CMM_hidden_dim)

        self.match_module = MatchModule(self.args.model.CMM_hidden_dim)
        self.multi_head_module = MultiHeadModule(self.args.model.head_num, self.args.model.weight)

        self.mclet_text_loss = MCLETLoss(embedding_dim=self.args.model.CMM_hidden_dim, cl_temperature=0.6)
        self.mclet_image_loss = MCLETLoss(embedding_dim=self.args.model.CMM_hidden_dim, cl_temperature=0.6)

    def forward(self,
                entity_text_cls, entity_text_tokens,
                entity_image_cls, entity_image_tokens,
                mention_text_cls, mention_text_tokens,
                mention_image_cls, mention_image_tokens):
        entity_text_cls = self.text_fc(entity_text_cls)
        entity_text_cls_ori = entity_text_cls
        entity_text_tokens = self.text_fc(entity_text_tokens)
        entity_image_cls = self.image_fc(entity_image_cls)
        entity_image_cls_ori = entity_image_cls
        entity_image_tokens = self.image_fc(entity_image_tokens)

        mention_text_cls = self.text_fc(mention_text_cls)
        mention_text_cls_ori = mention_text_cls
        mention_text_tokens = self.text_fc(mention_text_tokens)
        mention_image_cls = self.image_fc(mention_image_cls)
        mention_image_cls_ori = mention_image_cls
        mention_image_tokens = self.image_fc(mention_image_tokens)

        entity_text_cls = self.match_module([entity_text_cls_ori.unsqueeze(dim=1), entity_image_tokens]).squeeze()
        entity_image_tokens = self.match_module([entity_image_tokens, entity_text_cls_ori.unsqueeze(dim=1)])
        entity_text_image_context = self.multi_head_module(torch.cat([entity_text_cls.unsqueeze(dim=1), entity_image_tokens], dim=1))
        entity_text_image_gate_score = self.gate_act(self.gate_fc(entity_text_cls_ori))
        entity_text_image_context = self.gate_layer_norm((entity_text_cls_ori * entity_text_image_gate_score) + entity_text_image_context)

        entity_image_cls = self.match_module([entity_image_cls_ori.unsqueeze(dim=1), entity_text_tokens]).squeeze()
        entity_text_tokens = self.match_module([entity_text_tokens, entity_image_cls_ori.unsqueeze(dim=1)])
        entity_image_text_context = self.multi_head_module(torch.cat([entity_image_cls.unsqueeze(dim=1), entity_text_tokens], dim=1))
        entity_image_text_gate_score = self.gate_act(self.gate_fc(entity_image_cls_ori))
        entity_image_text_context = self.gate_layer_norm((entity_image_cls_ori * entity_image_text_gate_score) + entity_image_text_context)

        mention_text_cls = self.match_module([mention_text_cls_ori.unsqueeze(dim=1), mention_image_tokens]).squeeze()
        mention_image_tokens = self.match_module([mention_image_tokens, mention_text_cls_ori.unsqueeze(dim=1)])
        mention_text_image_context = self.multi_head_module(torch.cat([mention_text_cls.unsqueeze(dim=1), mention_image_tokens], dim=1))
        mention_text_image_gate_score = self.gate_act(self.gate_fc(mention_text_cls_ori))
        mention_text_image_context = self.gate_layer_norm((mention_text_cls_ori * mention_text_image_gate_score) + mention_text_image_context)

        mention_image_cls = self.match_module([mention_image_cls_ori.unsqueeze(dim=1), mention_text_tokens]).squeeze()
        mention_text_tokens = self.match_module([mention_text_tokens, mention_image_cls_ori.unsqueeze(dim=1)])
        mention_image_text_context = self.multi_head_module(torch.cat([mention_image_cls.unsqueeze(dim=1), mention_text_tokens], dim=1))
        mention_image_text_gate_score = self.gate_act(self.gate_fc(mention_image_cls_ori))
        mention_image_text_context = self.gate_layer_norm((mention_image_cls_ori * mention_image_text_gate_score) + mention_image_text_context)

        score_text_image_context = torch.matmul(mention_text_image_context, entity_text_image_context.transpose(-1, -2))
        score_image_text_context = torch.matmul(mention_image_text_context, entity_image_text_context.transpose(-1, -2))

        return score_text_image_context, score_image_text_context

class M3ELMatcher(nn.Module):
    def __init__(self, args):
        super(M3ELMatcher, self).__init__()
        self.args = args
        self.timm = TextIntraModalMatch(self.args)
        self.iimm = ImageIntraModalMatch(self.args)
        self.cmm = CrossModalMatch(self.args)
        self.cmmb = CrossModalMatchBidirection(self.args)

        self.text_cls_layernorm = nn.LayerNorm(self.args.model.dt)
        self.text_tokens_layernorm = nn.LayerNorm(self.args.model.dt)
        self.image_cls_layernorm = nn.LayerNorm(self.args.model.dv)
        self.image_tokens_layernorm = nn.LayerNorm(self.args.model.dv)

        self.text_fc = nn.Linear(self.args.model.input_hidden_dim, self.args.model.TIMM_hidden_dim)
        self.image_fc = nn.Linear(self.args.model.dv, self.args.model.IIMM_hidden_dim)
        self.scale_text_cls_layernorm = nn.LayerNorm(self.args.model.TIMM_hidden_dim)
        self.scale_text_tokens_layernorm = nn.LayerNorm(self.args.model.TIMM_hidden_dim)
        self.scale_image_cls_layernorm = nn.LayerNorm(self.args.model.IIMM_hidden_dim)
        self.scale_image_tokens_layernorm = nn.LayerNorm(self.args.model.IIMM_hidden_dim)

        self.mclet_text_loss = MCLETLoss(embedding_dim=self.args.model.IIMM_hidden_dim, cl_temperature=0.6)
        self.mclet_image_loss = MCLETLoss(embedding_dim=self.args.model.IIMM_hidden_dim, cl_temperature=0.6)
        self.weight_cl_loss = WeightedContrastiveLoss(temperature=self.args.model.loss_temperature, inter_weight=self.args.model.inter_weight, intra_weight=self.args.model.intra_weight)

    def forward(self,
                entity_text_cls, entity_text_tokens,
                mention_text_cls, mention_text_tokens,
                entity_image_cls, entity_image_tokens,
                mention_image_cls, mention_image_tokens,
                train_flag=False, bidirection=False):
        """

        :param entity_text_cls:     [num_entity, dim]
        :param entity_text_tokens:  [num_entity, max_seq_len, dim]
        :param mention_text_cls:    [batch_size, dim]
        :param mention_text_tokens: [batch_size, max_sqe_len, dim]
        :param entity_image_cls:    [num_entity, dim]
        :param mention_image_cls:   [batch_size, dim]
        :param entity_image_tokens: [num_entity, num_patch, dim]
        :param mention_image_tokens:[num_entity, num_patch, dim]
        :return:
        """
        if train_flag == True:
            text_cl_loss = self.weight_cl_loss(entity_text_cls, mention_text_cls)
            image_cl_loss = self.weight_cl_loss(entity_image_cls, mention_image_cls)
            cl_loss = (text_cl_loss + image_cl_loss) / 2
        else:
            cl_loss = None

        entity_text_cls = self.text_cls_layernorm(entity_text_cls)
        mention_text_cls = self.text_cls_layernorm(mention_text_cls)

        entity_text_tokens = self.text_tokens_layernorm(entity_text_tokens)
        mention_text_tokens = self.text_tokens_layernorm(mention_text_tokens)

        entity_image_cls = self.image_cls_layernorm(entity_image_cls)
        mention_image_cls = self.image_cls_layernorm(mention_image_cls)

        entity_image_tokens = self.image_tokens_layernorm(entity_image_tokens)
        mention_image_tokens = self.image_tokens_layernorm(mention_image_tokens)

        text_matching_score = self.timm(entity_text_cls, entity_text_tokens, mention_text_cls, mention_text_tokens)
        image_matching_score = self.iimm(entity_image_cls, entity_image_tokens, mention_image_cls, mention_image_tokens)

        if bidirection == True:
            text_image_matching_score, image_text_matching_score = self.cmmb(entity_text_cls, entity_text_tokens, entity_image_cls, entity_image_tokens, mention_text_cls, mention_text_tokens, mention_image_cls, mention_image_tokens)
            score = (text_matching_score + image_matching_score + text_image_matching_score + image_text_matching_score) / 4
            return score, (text_matching_score, image_matching_score, text_image_matching_score, image_text_matching_score), (cl_loss)
        else:
            image_text_matching_score = self.cmm(entity_text_cls, entity_image_tokens, mention_text_cls, mention_image_tokens)
            score = (text_matching_score + image_matching_score + image_text_matching_score) / 3
            return score, (text_matching_score, image_matching_score, image_text_matching_score), (cl_loss)
