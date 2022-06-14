import torch
import transformers

from BERT_explainability.modules.layers_ours import Linear, Sigmoid, Dropout, Softmax
from BERT_explainability.modules.BERT.BERT import BertModel
from BERT_explainability.modules.BERT.Electra import ElectraClassificationHead


def load_model():
    config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-uncased')
    model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased', config=config)
    return model


class BERT512(torch.nn.Module):

    def __init__(self):
        super(BERT512, self).__init__()
        self.bert = load_model()
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask, training=True):
        res = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(res.pooler_output)
        x = self.classifier(x)

        return x if training else self.sigmoid(x)


class BERT512ForCaptum(torch.nn.Module):

    def __init__(self):
        super(BERT512ForCaptum, self).__init__()
        self.bert = load_model()
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs_embeds, attention_mask):
        res = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        x = self.dropout(res.pooler_output)
        x = self.classifier(x)

        return self.activation_sigmoid(x)

    def get_embedding_matrix(self):
        return self.bert.base_model.embeddings.word_embeddings.weight.data


class BERT512RelProp(torch.nn.Module):
    def __init__(self):
        super(BERT512RelProp, self).__init__()
        config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-uncased')
        self.bert = BertModel(config)
        self.dropout = Dropout(0.1)
        self.classifier = Linear(768, 1)
        self.sigmoid = Sigmoid()

    def forward(self, input_ids, attention_mask):
        res = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(res.pooler_output)
        x = self.classifier(x)
        return self.sigmoid(x)

    def relprop(self, cam=None, **kwargs):
        cam = self.sigmoid.relprop(cam, **kwargs)
        cam = self.classifier.relprop(cam, **kwargs)
        cam = self.dropout.relprop(cam, **kwargs)
        cam = self.bert.relprop(cam, **kwargs)
        return cam


class BertSequenceClassifierSST(transformers.BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = Dropout(classifier_dropout)
        self.classifier = Linear(config.hidden_size, config.num_labels)
        self.softmax = Softmax(dim=-1)

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        inputs_embeds_in_input_ids=True
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids if not inputs_embeds_in_input_ids else None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds if not inputs_embeds_in_input_ids else input_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if return_logits:
            return logits
        else:
            return self.softmax(logits)

    def relprop(self, cam=None, **kwargs):
        cam = self.classifier.relprop(cam, **kwargs)
        cam = self.dropout.relprop(cam, **kwargs)
        cam = self.bert.relprop(cam, **kwargs)
        # print("conservation: ", cam.sum())
        return cam


class BertSequenceClassifierNews(transformers.BertForSequenceClassification, transformers.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = Dropout(classifier_dropout)
        self.classifier = Linear(config.hidden_size, config.num_labels)
        self.sigmoid = Sigmoid()

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        inputs_embeds_in_input_ids=True
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids if not inputs_embeds_in_input_ids else None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds if not inputs_embeds_in_input_ids else input_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if return_logits:
            return logits
        else:
            return self.sigmoid(logits)


    def relprop(self, cam=None, **kwargs):
        cam = self.classifier.relprop(cam, **kwargs)
        cam = self.dropout.relprop(cam, **kwargs)
        cam = self.bert.relprop(cam, **kwargs)
        # print("conservation: ", cam.sum())
        return cam


class RobertaSequenceClassifierNews(transformers.RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.sigmoid = Sigmoid()
        self.dropout = Dropout(config.classifier_dropout)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        inputs_embeds_in_input_ids=True
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids=input_ids if not inputs_embeds_in_input_ids else None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds if not inputs_embeds_in_input_ids else input_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[0]

        logits = self.classifier(pooled_output)

        if return_logits:
            return logits
        else:
            return self.sigmoid(logits)


class ElectraSequenceClassifierNews(transformers.ElectraForSequenceClassification, transformers.ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.electra = transformers.ElectraModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = Dropout(classifier_dropout)
        self.classifier = ElectraClassificationHead(config)
        self.sigmoid = Sigmoid()

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        inputs_embeds_in_input_ids=True
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.electra(
            input_ids=input_ids if not inputs_embeds_in_input_ids else None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds if not inputs_embeds_in_input_ids else input_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        if return_logits:
            return logits
        else:
            return self.sigmoid(logits)

