import torch
from torch import nn
import torch.nn.functional as F


class Baseline(nn.Module):

    def __init__(self, encoder, output_dim, encoder_type):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = encoder.config.hidden_size
        self.output_dim = output_dim
        self.encoder_type = encoder_type
        self.dropout = nn.Dropout(p=0.1)
        # self.bn = nn.BatchNorm1d(self.hidden_size)
        
        #self.ln = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.out = nn.Linear(self.hidden_size, self.output_dim)
        #self.bn = nn.BatchNorm1d(self.output_dim, eps=1e-12)
        #self.prelu = nn.PReLU()

    def forward(self, input_ids, attention_mask, token_type_ids):
        '''
        :param encoder_type: encoder_type:  "first-last-avg", "last-avg", "cls", "pooler(cls + dense)"
        :return:
        '''
        encoder_type = self.encoder_type
        output = self.encoder(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if encoder_type == 'first-last-avg':
            # 第一层和最后一层的隐层取出  然后经过平均池化
            first = output.hidden_states[1]  # hidden_states列表有13个hidden_state，第一个其实是embeddings，第二个元素才是第一层的hidden_state
            last = output.hidden_states[-1]
            seq_length = first.size(1)  # 序列长度

            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            final_encoding = torch.avg_pool1d(
                torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2),
                kernel_size=2).squeeze(-1)
            out = final_encoding

        elif encoder_type == 'last-avg':
            sequence_output = output.last_hidden_state  # (batch_size, max_len, hidden_size)
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            out = final_encoding

        elif encoder_type == "cls":
            sequence_output = output.last_hidden_state
            cls = sequence_output[:, 0]  # [b,d]
            out = cls

        elif encoder_type == "pooler":
            pooler_output = output.pooler_output  # [b,d]
            out = pooler_output
            
        elif encoder_type == "last3-avg":
            all_feats=[]
            for feat in output.hidden_states[-3:]:
                seq_length = feat.size(1)
                final_encoding = torch.avg_pool1d(feat.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
                all_feats.append(final_encoding)
            # out=torch.cat(all_feats, dim=1)
            out = torch.avg_pool1d(torch.cat([all_feats[0].unsqueeze(1), all_feats[1].unsqueeze(1), all_feats[2].unsqueeze(1)], dim=1).transpose(1, 2), kernel_size=3).squeeze(-1)
            # out = torch.nn.functional.normalize(out.float(),dim=1).to(self.device)

        out = self.dropout(out)
        #out = self.ln(out)
        out = self.out(out)
        #out = self.bn(out)
        #out = self.prelu(out)
        return out
    
    
