from torch import nn
from transformers import BertModel


class ClassifierModel(nn.Module):
    def __init__(self,
                 bert_dir,
                 n_class,
                 dropout_prob=0.1):
        super(ClassifierModel, self).__init__()
        self.bert_module = BertModel.from_pretrained(bert_dir)
        self.bert_config = self.bert_module.config
        self.dropout_layer = nn.Dropout(dropout_prob)
        out_dims = self.bert_config.hidden_size
        self.obj_classifier = nn.Sequential(
            nn.Linear(out_dims, 64),
            nn.ReLU(),
            nn.Linear(64, n_class)
        )

    def forward(self,
                input_ids,
                input_mask,
                segment_ids,
                label_id=None):

        bert_outputs = self.bert_module(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids
        )
        last_hidden_state_cls = bert_outputs[0][:, 0, :]
        last_hidden_state_cls = self.dropout_layer(last_hidden_state_cls)
        out = self.obj_classifier(last_hidden_state_cls)
        return out
