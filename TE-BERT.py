
class TE_BERT(nn.Module):
    def __init__(self, mode, bow_len, lr, lr_alpha, warm, topic_num, prior_a):
        super().__init__()

        self.model = AutoModel.from_pretrained('bert-base-uncased').to(device)

        self.classifier = nn.Sequential(
            nn.Linear((768 + topic_num)*3, 2)
        ).to(device)

        # encoder_1
        self.encoder = nn.Sequential( 
            nn.Linear(bow_len, 10000),
            nn.BatchNorm1d(10000),
            nn.Softplus(),
            nn.Dropout(p=0.2),
            
            nn.Linear(10000,5000),
            nn.Softplus(),

            nn.Linear(5000,2500),
            nn.BatchNorm1d(2500), 
            nn.Softplus(),
            
            nn.Linear(2500,1250),
            nn.Softplus(),   
            nn.Dropout(p=0.2),
            
            nn.Linear(1250,500),
            nn.BatchNorm1d(500), 
            nn.Softplus(),
            
            nn.Linear(500, topic_num), # 200
            nn.BatchNorm1d(topic_num)
        ).to(device)


        self.decoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(topic_num, bow_len),
            nn.BatchNorm1d(bow_len)
        ).to(device)

        
        self.prior_a = prior_a # prior_a = 1
        self.prior_alpha = nn.Parameter(torch.ones(topic_num).to(device) * self.prior_a)

        
        self.optimizer = torch.optim.AdamW(
           [
                {'params': self.model.parameters(), 'lr': 1e-5},
                {'params': self.classifier.parameters(), 'lr' : 1e-4},
                {'params': self.encoder.parameters() , 'lr' : lr},
                {'params': self.decoder.parameters(), 'lr': lr},
                {'params': self.prior_alpha, 'lr': lr_alpha}
           ]
        )

        
        self.early_stop = early_stopping(patience, delta) # 5, 0.005
        self.criterion = nn.CrossEntropyLoss()
        
        self.mode = mode

        self.warm_up = warm
    
    def forward(self, ids1, att1, ids2, att2, bow1=None, bow2=None, label=None):
        output1 = self.model(
            input_ids = ids1,
            attention_mask = att1
        ).last_hidden_state

        output2 = self.model(
            input_ids = ids2,
            attention_mask = att2
        ).last_hidden_state

        if self.mode == 'cls':
            output1_cls = output1[:, 0, :]
            output2_cls = output2[:, 0, :]

            return output1_cls, output2_cls

        if self.mode == 'mean':
            output1_mean = torch.mean(output1, dim = 1)
            output2_mean = torch.mean(output2, dim = 1)

            return output1_mean, output2_mean


    def output_cat(self, result1 , result2, dist1, dist2):

        temp1 = torch.cat([result1, dist1], dim = 1)
        temp2 = torch.cat([result2, dist2], dim = 1)
        
        last_embedding = torch.cat([temp1, temp2, torch.abs(temp1 - temp2)], dim = 1)
        logits = self.classifier(last_embedding)
        prob = F.softmax(logits, dim = 1)
        
        return logits, prob

    
    def encoder_foward(self, bow1, bow2):      

        sum_bow1 = bow1.sum(dim = -1, keepdim=True)
        sum_bow2 = bow2.sum(dim = -1, keepdim=True)

        normalized_bow1 = bow1 / sum_bow1 
        normalized_bow2 = bow2 / sum_bow2

        encoder_result1 = self.encoder(normalized_bow1)
        encoder_result2 = self.encoder(normalized_bow2)

        alpha1 = F.softplus(encoder_result1) + 1e-3
        alpha2 = F.softplus(encoder_result2) + 1e-3    

        dist1 = Dirichlet(alpha1).rsample()
        dist2 = Dirichlet(alpha2).rsample()

        return Dirichlet(alpha1), Dirichlet(alpha2), dist1, dist2

    
    def decode(self, theta):
        beta_logit = self.decoder(theta)
        beta = torch.log_softmax(beta_logit, dim = 1)

        return beta


    def reconstruct_loss(self, input, output):
        input_norm = input / input.sum(dim=1, keepdim=True)
        nll = -torch.sum(input_norm*output)

        return nll

    
    def KL_divergence(self, posterior):
        
        self.prior = Dirichlet(F.softplus(self.prior_alpha))
        kl_loss = kl_divergence(posterior, self.prior)

        return kl_loss.mean()


    def kl_anealing(self, epoch, warm_up, max_beta = 1.0):
        
        if epoch >= warm_up:
            return max_beta 

        else:
            return max_beta * epoch / warm_up
