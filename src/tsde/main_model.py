import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from base.denoisingNetwork import diff_Block
from base.mtsEmbedding import embedding_MTS

from utils.masking_strategies import get_mask_probabilistic_layering, get_mask_equal_p_sample, imputation_mask_batch, pattern_mask_batch, interpolation_mask_batch


class TSDE_base(nn.Module):
    """
    Base class for TSDE model.
    
    Attributes:
        - device: The device on which the model will run (CPU or CUDA).
        - target_dim: number of features in the MTS.
        - sample_feat: Whether to sample subset of features during training.
        - mix_masking_strategy: Strategy for mixing masks during pretraining.
        - time_strategy: Strategy for embedding time points.
        - emb_time_dim: Dimension of time embeddings.
        - emb_cat_feature_dim: Dimension of categorical feature embeddings.
        - mts_emb_dim: Dimension of the MTS embeddings.
        - embed_layer: Embedding layer for feature embeddings.
        - diffmodel: Model block for diffusion.
        - embdmodel: Model for embedding MTS.
        - mlp: Multi-layer perceptron for classification tasks.
        - conv: Convolutional layer for anomaly detection.
        
    Methods:
        - time_embedding: Generates sinusoidal embeddings for time points.
        - get_mts_emb: Generates embeddings for MTS.
        - calc_loss: Calculates training loss for a given batch of data.
        - calc_loss_valid: Calculates validation loss for a given batch of data.
        - impute: Imputes missing values in the time series.
        - forward: Forward pass for pretraining, and fine-tuning for imputation, interpolation, and forecasting.
        - forward_finetuning: Forward pass for fine-tuning on specific tasks (classification or anomaly detection).
        - evaluate_finetuned_model: Evaluates the fine-tuned model for classification and anomaly detection.
        - evaluate: Evaluates the model on imputation, interpolation and forecasting.
    """
    def __init__(self, target_dim, config, device, sample_feat):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.sample_feat=sample_feat

        self.mix_masking_strategy = config["model"]["mix_masking_strategy"]   
        self.time_strategy = config["model"]["time_strategy"]
        
        self.emb_time_dim = config["embedding"]["timeemb"]
        self.emb_cat_feature_dim = config["embedding"]["featureemb"]  
        
        self.mts_emb_dim = 1+2*config["embedding"]["channels"]
        
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_cat_feature_dim
        )
        
        config_diff = config["diffusion"]
        config_diff["mts_emb_dim"] = self.mts_emb_dim
        config_emb = config["embedding"]

        

        self.diffmodel = diff_Block(config_diff)
        self.embdmodel = embedding_MTS(config_emb)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat) 
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        
        L = config_emb["num_timestamps"]
        K = config_emb["num_feat"]

        # Number of classes for classification experiments
        num_classes = config_emb["classes"]
        
        ## Classifier head
        self.mlp = nn.Sequential(
            nn.Linear(L*K*self.mts_emb_dim, 256),  # Adjust as necessary
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        
        ## projection to reconstruct MTS for Anomaly Detection
        self.conv = nn.Linear((self.mts_emb_dim-1)*K, K, bias=True)

        
    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    
    def get_mts_emb(self, observed_tp, cond_mask, x_co, feature_id):
        B, K, L = cond_mask.shape
        if self.time_strategy == "hawkes":

            time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        elif self.time_strategy == "categorical embeddings":
            time_embed = self.time_embed_layer(
                torch.arange(L).to(self.device)
            )  # (L,emb)
            time_embed = time_embed.unsqueeze(0).expand(B, -1, -1) ### (B,L,128)

        if feature_id is None:
            feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
            ) 
            feature_embed = feature_embed.unsqueeze(0).expand(B, -1, -1)
        else:
            feature_embed = self.embed_layer(
            feature_id
            )  # (K,emb)         
        #print(x_co.shape, time_embed.shape, feature_embed.shape)
        cond_embed, xt, xf = self.embdmodel(x_co, time_embed, feature_embed)
        
        side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
        mts_emb = torch.cat([cond_embed, side_mask], dim=1)
        
        return mts_emb


    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, mts_emb, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, mts_emb, is_train, set_t=t
            )
            loss_sum += loss.detach()
               
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, mts_emb, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, mts_emb, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)

        return noisy_target

    def impute(self, observed_data, cond_mask, mts_emb, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        
        for i in range(n_samples):

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                
                predicted= self.diffmodel(noisy_target, mts_emb, torch.tensor([t]).to(self.device))
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1, task='pretraining', normalize_for_ad=False):
        ## is_train = 1 for pretraining and for finetuning but task should be specified and = 0 for evaluation
        
        (
            observed_data,
            observed_mask,
            feature_id,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
            _,
            _
        ) = self.process_data(batch, sample_feat=self.sample_feat, train=is_train)
        if is_train == 0:
            cond_mask = gt_mask
        else:
            if task == 'pretraining':
                if self.mix_masking_strategy == 'equal_p':
                    cond_mask = get_mask_equal_p_sample(observed_mask)
                elif self.mix_masking_strategy == 'probabilistic_layering':
                    cond_mask = get_mask_probabilistic_layering(observed_mask)
                else:
                    print('Please choose one of the following masking strategy in the config: equal_p, probabilistic_layering')
            elif task == 'Imputation':
                cond_mask = imputation_mask_batch(observed_mask)
            elif task == 'Interpolation':
                cond_mask = interpolation_mask_batch(observed_mask)
            elif task == 'Imputation with pattern':
                cond_mask = pattern_mask_batch(observed_mask)
            elif task == 'Forecasting':
                cond_mask = gt_mask
            else:
                print('Please choose the right masking to be applied during finetuning')

        if normalize_for_ad:
            ## Normalization from non-stationary Transformer
            means = observed_data.mean(2, keepdim=True)
            observed_data = observed_data-means
            stdev = torch.sqrt(torch.var(observed_data, dim=2, keepdim=True, unbiased=False) + 1e-5)
            observed_data /= stdev


        x_co = (cond_mask * observed_data).unsqueeze(1)    
        mts_emb = self.get_mts_emb(observed_tp, cond_mask, x_co, feature_id)
        
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, mts_emb, is_train)

    def forward_finetuning(self, batch, criterion, task='classification', normalize_for_ad=False):
        ## task should be either, classification or anomaly_detection
        
        (
            observed_data,
            observed_mask,
            feature_id,
            observed_tp,
            gt_mask,
            _,
            _,
            _,
            classes
        ) = self.process_data(batch, sample_feat=self.sample_feat, train=False)
        
        if normalize_for_ad:
            ## Normalization from non-stationary Transformer
            original_observed_data = observed_data.clone()
            means = observed_data.mean(2, keepdim=True)
            observed_data = observed_data-means
            stdev = torch.sqrt(torch.var(observed_data, dim=2, keepdim=True, unbiased=False) + 1e-5)
            observed_data /= stdev

        x_co = (observed_mask * observed_data).unsqueeze(1)    
        mts_emb = self.get_mts_emb(observed_tp, observed_mask, x_co, feature_id)
        
        if task == 'classification':
            outputs = self.mlp(mts_emb.reshape(mts_emb.shape[0],-1)) 
            classes = classes.to(self.device)
            loss = criterion(outputs, classes)
            return outputs, loss
        elif task == 'anomaly_detection':
            B, C, K, L =mts_emb.shape
            #outputs = self.projection(mts_emb.permute(0,2,3,1)).squeeze(-1)
            outputs = self.conv(mts_emb[:, :C-1, :, :].reshape(B, (C-1)*K, L).permute(0,2,1)).permute(0,2,1)
            if normalize_for_ad:
                dec_out = outputs * \
                      (stdev[:, :, 0].unsqueeze(2).repeat(
                          1, 1, L))
                outputs = dec_out + \
                      (means[:, :, 0].unsqueeze(2).repeat(
                          1, 1, L))

            loss = criterion(outputs, original_observed_data)
            return outputs, loss
        
    def evaluate_finetuned_model(self, batch, criterion= nn.MSELoss(reduction='none'), task='classification', normalize_for_ad=False):
        
        (
            observed_data,
            observed_mask,
            feature_id,
            observed_tp,
            gt_mask,
            _,
            _,
            _,
            classes
        ) = self.process_data(batch, sample_feat=self.sample_feat, train=False)
        
        with torch.no_grad():

            if normalize_for_ad:
                ## Normalization from non-stationary Transformer
                original_observed_data = observed_data.clone()
                means = observed_data.mean(2, keepdim=True)
                observed_data = observed_data-means
                stdev = torch.sqrt(torch.var(observed_data, dim=2, keepdim=True, unbiased=False) + 1e-5)
                observed_data /= stdev

            x_co = (observed_mask * observed_data).unsqueeze(1)    
            mts_emb = self.get_mts_emb(observed_tp, observed_mask, x_co, feature_id)

            if task == 'classification':
                outputs = self.mlp(mts_emb.reshape(mts_emb.shape[0],-1)) 
                classes = classes.to(self.device)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted_classes = torch.max(probabilities, 1)
                correct = (predicted_classes == torch.tensor(classes)).sum().item()
                total = classes.size(0)
                accuracy = correct/total
                #print(probabilities.cpu().numpy())
                #auc = roc_auc_score(classes.cpu().numpy(), probabilities[:, 1].cpu().numpy())
                #print('AUC', auc)
                return (outputs, classes), (correct, total)
            elif task == 'anomaly_detection':
                B, C, K, L =mts_emb.shape
                outputs = self.conv(mts_emb[:, :C-1, :, :].reshape(B, (C-1)*K, L).permute(0,2,1)).permute(0,2,1)
                #outputs = self.projection(mts_emb.permute(0,2,3,1)).squeeze(-1) 
                if normalize_for_ad:
                    dec_out = outputs * \
                        (stdev[:, :, 0].unsqueeze(2).repeat(
                            1, 1, L))
                    outputs = dec_out + \
                        (means[:, :, 0].unsqueeze(2).repeat(
                            1, 1, L))
                score = torch.mean(criterion(original_observed_data, outputs), dim=1)
                score = score.detach().cpu().numpy()
                return outputs, score

        
        
    def evaluate(self, batch, n_samples, normalize_for_ad=False):
        (
            observed_data,
            observed_mask,
            feature_id,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            _,
            labels
        ) = self.process_data(batch, sample_feat=self.sample_feat, train=False)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            if normalize_for_ad:
                ## Normalization from non-stationary Transformer
                original_observed_data = observed_data.clone()
                means = observed_data.mean(2, keepdim=True)
                observed_data = observed_data-means
                stdev = torch.sqrt(torch.var(observed_data, dim=2, keepdim=True, unbiased=False) + 1e-5)
                observed_data /= stdev

            x_co = (cond_mask * observed_data).unsqueeze(1)
            mts_emb = self.get_mts_emb(observed_tp, cond_mask, x_co, feature_id)

            samples = self.impute(observed_data, cond_mask, mts_emb, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        if normalize_for_ad:
            if labels is not None:
                return samples, original_observed_data, target_mask, observed_mask, observed_tp, labels
            else:
                return samples, original_observed_data, target_mask, observed_mask, observed_tp
        else:
            if labels is not None:
                return samples, observed_data, target_mask, observed_mask, observed_tp, labels
            else:
                return samples, observed_data, target_mask, observed_mask, observed_tp
    


class TSDE_Forecasting(TSDE_base):
    """
    Specialized TSDE model for forecasting tasks.
    
    This class extends the TSDE_base model to specifically handle forecasting by processing the input data appropriately.
    """
    def __init__(self, config, device, target_dim, sample_feat=False):
        super(TSDE_Forecasting, self).__init__(target_dim, config, device, sample_feat)

    def process_data(self, batch, sample_feat, train=True):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        feature_id = batch["feature_id"].to(self.device).long()
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        if train and sample_feat:
            sampled_data = []
            sampled_mask = []
            sampled_feature_id = []
            sampled_gt_mask = []
            size = 128
            
            for i in range(len(observed_data)):
                ind = np.arange(feature_id.shape[1])
                np.random.shuffle(ind)
                sampled_data.append(observed_data[i,ind[:size],:])
                sampled_mask.append(observed_mask[i,ind[:size],:])
                sampled_feature_id.append(feature_id[i,ind[:size]])
                sampled_gt_mask.append(gt_mask[i,ind[:size],:])
            observed_data = torch.stack(sampled_data,0)
            observed_mask = torch.stack(sampled_mask,0)
            feature_id = torch.stack(sampled_feature_id,0)
            gt_mask = torch.stack(sampled_gt_mask,0)
        
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            feature_id,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            None,
            None,
        )


class TSDE_PM25(TSDE_base):
    """
    Specialized TSDE model for PM2.5 environmental data.
    
    Designed to handle and process PM2.5 data for imputation.
    """
    def __init__(self, config, device, target_dim=36, sample_feat=False):
        super(TSDE_PM25, self).__init__(target_dim, config, device, sample_feat)

    def process_data(self, batch, train, sample_feat):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            None,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            None,
            None,
        )


class TSDE_Physio(TSDE_base):
    """
    Specialized TSDE model for PhysioNet dataset.
    
    Adapts the TSDE_base model for tasks involving PhysioNet data, including imputation and interpolation.
    """
    def __init__(self, config, device, target_dim=35, sample_feat=False):
        super(TSDE_Physio, self).__init__(target_dim, config, device, sample_feat)

    def process_data(self, batch, train, sample_feat):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        labels = batch["labels"]
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            None,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            None,
            labels,

        )


class TSDE_AD(TSDE_base):
    """
    Specialized TSDE model for anomaly detection datasets.
    
    Tailors the TSDE_base model for anomaly detection datasets, including MSL, SMD, PSM, SMAP and SWaT.
    """
    def __init__(self, config, device, target_dim=55, sample_feat=False):
        super(TSDE_AD, self).__init__(target_dim, config, device, sample_feat)

    def process_data(self, batch, train, sample_feat):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        label = batch["label"].to(self.device).float()
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask
        return (
            observed_data,
            observed_mask,
            None,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            None,
            label,
        )    
    
