import torch
import torch.autograd as autograd
import torch.nn as nn
import math
import contextlib
from .Model import Model

class MANoR(Model):
    def __init__(
        self,
        ent_tot,
        rel_tot,
        dim=100,
        margin=6.0,
        epsilon=2.0,
        img_emb=None,
        text_emb=None
    ):
        super(MANoR, self).__init__(ent_tot, rel_tot)
        assert img_emb is not None
        assert text_emb is not None

        self.margin = margin
        self.epsilon = epsilon
        self.dim_e = dim * 2
        self.dim_r = dim
        self.num_heads = 4
        self.head_dim = self.dim_e // self.num_heads

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)

        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),
            requires_grad=False
        )
        # Pre-trained Modality Features
        self.img_dim = img_emb.shape[1]
        self.text_dim = text_emb.shape[1]
        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(False)
        self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(False)
        
        bottleneck_hidden = max(256, self.dim_e // 4)
        self.G_ent = nn.Sequential(
            nn.Linear(self.dim_e, bottleneck_hidden),
            nn.ReLU(),
            nn.Linear(bottleneck_hidden, self.dim_e)
        )
        self.G_img = nn.Sequential(
            nn.Linear(self.dim_e, bottleneck_hidden),
            nn.ReLU(),
            nn.Linear(bottleneck_hidden, self.dim_e)
        )
        self.G_text = nn.Sequential(
            nn.Linear(self.dim_e, bottleneck_hidden),
            nn.ReLU(),
            nn.Linear(bottleneck_hidden, self.dim_e)
        )
        
        self.img_proj = nn.Sequential(
            nn.Linear(self.img_dim, self.dim_e),
            nn.ReLU(),
            nn.Linear(self.dim_e, self.dim_e)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.dim_e),
            nn.ReLU(),
            nn.Linear(self.dim_e, self.dim_e)
        )
        
        # Multi-head layers
        self.fusion_layers = nn.ModuleList([
            nn.Linear(2 * self.head_dim, self.head_dim) for _ in range(self.num_heads)
        ])

        # Q/K/V projections per head
        self.q_proj = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(self.num_heads)])
        self.k_proj = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(self.num_heads)])
        self.v_proj = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(self.num_heads)])

        for q_layer, k_layer, v_layer in zip(self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(q_layer.weight.data)
            nn.init.xavier_uniform_(k_layer.weight.data)
            nn.init.xavier_uniform_(v_layer.weight.data)
            nn.init.zeros_(q_layer.bias)
            nn.init.zeros_(k_layer.bias)
            nn.init.zeros_(v_layer.bias)

        self.final_layer_norm = nn.LayerNorm(self.dim_e)

        self.V_global = nn.Linear(self.dim_e, 1, bias=False)
        self.V_global.requires_grad_(True)

        nn.init.uniform_(
            tensor=self.ent_embeddings.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]),
            requires_grad=False
        )
        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )

        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False

        self.rel_gate = nn.Embedding(self.rel_tot, 1)
        nn.init.uniform_(
            tensor=self.rel_gate.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )
 
        self._enhancement_topk = 1024        
        self._enhancement_temp = 0.01        
        self._noise_threshold = 0.001        
        self._use_amp_enhancement = True     
        self.rel_to_ent_proj = None

    def _safe_ids(self, ids_tensor, num, name):
        assert ids_tensor.dtype == torch.long, f"{name} dtype must be torch.long, got {ids_tensor.dtype}"
        if ids_tensor.min().item() < 0 or ids_tensor.max().item() >= num:
            ids_tensor.clamp_(min=0, max=num - 1)
        return ids_tensor

    def gated_fusion(self, emb, rel):
        # emb: batch_size x dim
        # rel: batch_size x dim
        w = torch.sigmoid(emb * rel)
        return w * emb + (1 - w) * rel

    def calibrated_fusion(self, es, ev, et, rg):
        # Stack initial features
        e = torch.stack((es, ev, et), dim=1)
        u = torch.tanh(e)
    
        modal_sim = torch.matmul(u, u.transpose(-2, -1)) / math.sqrt(u.size(-1))
        modal_affinity = torch.softmax(modal_sim, dim=-1)
    
        uncertainty_scores = torch.var(u, dim=-1, keepdim=True)
        uncertainty_weights = torch.softmax(-uncertainty_scores, dim=1)
    
        enhanced_u = torch.matmul(modal_affinity, u)
        u = u + uncertainty_weights * enhanced_u
    
        base_scores = self.V_global(u).squeeze(-1)      # [B,3]
        
        # Calibrate logits with attention entropy
        calibrated_scores = base_scores * (1 + torch.mean(modal_affinity, dim=-1))
        # Calculate Entropy for Temperature Bias
        with torch.no_grad():
            summaries = []
            refiners = [
                self.G_ent,
                self.G_img,
                self.G_text
            ]
            for i, refiner in enumerate(refiners):
                z = refiner(e[:, i, :])
                summaries.append(z)
            S = torch.stack(summaries, dim=1)           # [B,3,D]
    
            P = torch.softmax(S, dim=-1)
            entropy = -torch.sum(P * torch.log(P + 1e-9), dim=-1)  # [B,5]
            entropy = entropy / math.log(S.size(-1) + 1e-9)
    
            # entropy ∈ [0,1] → mild temperature bias
            temp_bias = 1.0 + entropy                   # no new hyperparam
    
        rel_gate = torch.sigmoid(rg)                    
        logits = calibrated_scores / (rel_gate * temp_bias + 1e-9)
    
        attention_weights = torch.softmax(logits, dim=-1)
        joint_vectors = torch.sum(
            attention_weights.unsqueeze(-1) * e, dim=1
        )
    
        return joint_vectors
    
    def cal_score(self, embs):
        return self._calc(embs[0], embs[2], embs[1], "")
        
    def _calc(self, h, t, r, mode):
        pi = self.pi_const
        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)
        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_head = re_head.view(-1,
                               re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1,
                               re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1,
                               re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1,
                               re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(
            -1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(
            -1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

        if mode == "head_batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        return score.permute(1, 0).flatten()
    
    def negotiation_gate_proj(self, head_idx, values, current_slice):
        # values/current_slice: [K, Hd]
        lin = self.fusion_layers[head_idx]
        W = lin.weight    # [Hd, 2*Hd]
        b = lin.bias
        Hd = values.size(-1)
        W1 = W[:, :Hd]    # [Hd, Hd]
        W2 = W[:, Hd:]    # [Hd, Hd]
        fused = values @ W1.T + current_slice @ W2.T
        if b is not None:
            fused = fused + b
        return fused  # [K, Hd]

    def _topk_softmax(self, scores, k, temperature):
        # scores: [1, K]
        k = min(k, scores.size(-1))
        topv, topi = torch.topk(scores, k, dim=-1)
        w = torch.softmax(topv / max(1e-6, temperature), dim=-1)  # [1,k]
        return w, topi
    
    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']

        
        batch_h = self._safe_ids(batch_h, self.ent_tot, "batch_h")
        batch_t = self._safe_ids(batch_t, self.ent_tot, "batch_t")
        batch_r = self._safe_ids(batch_r, self.rel_tot, "batch_r")

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        
        rg = self.rel_gate(batch_r)

        h_joint = self.calibrated_fusion(h, h_img_emb, h_text_emb, rg)
        t_joint = self.calibrated_fusion(t, t_img_emb, t_text_emb, rg)

        score = self.margin - self._calc(h_joint, t_joint, r, mode)
        return score

    def get_batch_ent_embs(self, data):
        return self.ent_embeddings(data)

    def get_batch_vis_embs(self, data):
        return self.img_proj(self.img_embeddings(data))

    def get_batch_text_embs(self, data):
        return self.text_proj(self.text_embeddings(data))

    def get_batch_ent_multimodal_embs(self, data):
        return self.ent_embeddings(data), self.img_proj(self.img_embeddings(data)), self.text_proj(self.text_embeddings(data))

    def get_fake_score(
        self,
        batch_h,
        batch_r,
        batch_t,
        mode,
        fake_hv=None,
        fake_tv=None,
        fake_ht=None,
        fake_tt=None
    ):
        if fake_hv is None or fake_tv is None or fake_ht is None or fake_tt is None:
            raise NotImplementedError

        
        batch_h = self._safe_ids(batch_h, self.ent_tot, "batch_h")
        batch_t = self._safe_ids(batch_t, self.ent_tot, "batch_t")
        batch_r = self._safe_ids(batch_r, self.rel_tot, "batch_r")

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        
        rg = self.rel_gate(batch_r)

        h_joint = self.calibrated_fusion(h, h_img_emb, h_text_emb, rg)
        t_joint = self.calibrated_fusion(t, t_img_emb, t_text_emb, rg)

        with torch.no_grad():
            S = min(1024, batch_h.size(0))
            batch_t_init = batch_t[:S]
            batch_h_init = batch_h[:S]
            h_init = h[:S]
            t_init = t[:S]
            h_joint_init = h_joint[:S]
            t_joint_init = t_joint[:S]

            num_heads = self.num_heads
            head_dim = self.head_dim
            temperature = self._enhancement_temp
            k_top = self._enhancement_topk

            # tails
            t_final = t_joint
            unique_tails = torch.unique(batch_t_init)
            for i in range(unique_tails.size(0)):
                tail_id = unique_tails[i]
                tail_indices = torch.where(batch_t_init == tail_id)[0]
                if tail_indices.numel() == 0:
                    continue

                related_h = h_init.index_select(0, tail_indices)
                related_h_joint = h_joint_init.index_select(0, tail_indices)

                first_idx = tail_indices[0]
                current_t = t_init[first_idx]
                original_t_joint = t_joint_init[first_idx]

                outputs = []
                amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if self._use_amp_enhancement else contextlib.nullcontext()
                with amp_ctx:
                    for head in range(num_heads):
                        s = head * head_dim; e = s + head_dim
                        q_in = current_t[s:e].unsqueeze(0)
                        k_in = related_h[:, s:e]
                        v_in = related_h_joint[:, s:e]

                        query = self.q_proj[head](q_in)                                  # [1,Hd]
                        keys = self.k_proj[head](k_in)                                   # [K,Hd]
                        scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(head_dim)  # [1,K]
                        w, idx = self._topk_softmax(scores, min(k_top, k_in.size(0)), temperature)  # [1,k]

                        values = self.v_proj[head](v_in)                                  # [K,Hd]
                        sel_idx = idx.squeeze(0)                                          # [k]
                        values_k = values.index_select(0, sel_idx)                        # [k,Hd]
                        cur_slice = original_t_joint[s:e].unsqueeze(0).expand(values_k.size(0), -1)  # [k,Hd]
                        fused_k = self.negotiation_gate_proj(head, values_k, cur_slice)              # [k,Hd]
                        head_out = torch.matmul(w, fused_k).squeeze(0)                    # [Hd]
                        outputs.append(head_out.float() if self._use_amp_enhancement else head_out)

                t_enhanced = torch.cat(outputs, dim=0)
                t_enhanced = self.final_layer_norm(t_enhanced + original_t_joint)
                t_final.index_copy_(0, tail_indices, t_enhanced.expand(tail_indices.size(0), -1))

            # heads
            h_final = h_joint
            unique_heads = torch.unique(batch_h_init)
            for i in range(unique_heads.size(0)):
                head_id = unique_heads[i]
                head_indices = torch.where(batch_h_init == head_id)[0]
                if head_indices.numel() == 0:
                    continue

                related_t = t_init.index_select(0, head_indices)
                related_t_joint = t_joint_init.index_select(0, head_indices)

                first_idx = head_indices[0]
                current_h = h_init[first_idx]
                original_h_joint = h_joint_init[first_idx]

                outputs = []
                amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if self._use_amp_enhancement else contextlib.nullcontext()
                with amp_ctx:
                    for head in range(num_heads):
                        s = head * head_dim; e = s + head_dim
                        q_in = current_h[s:e].unsqueeze(0)
                        k_in = related_t[:, s:e]
                        v_in = related_t_joint[:, s:e]

                        query = self.q_proj[head](q_in)
                        keys = self.k_proj[head](k_in)
                        scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(head_dim)
                        w, idx = self._topk_softmax(scores, min(k_top, k_in.size(0)), temperature)

                        values = self.v_proj[head](v_in)
                        sel_idx = idx.squeeze(0)
                        values_k = values.index_select(0, sel_idx)
                        cur_slice = original_h_joint[s:e].unsqueeze(0).expand(values_k.size(0), -1)
                        fused_k = self.negotiation_gate_proj(head, values_k, cur_slice)
                        head_out = torch.matmul(w, fused_k).squeeze(0)
                        outputs.append(head_out.float() if self._use_amp_enhancement else head_out)

                h_enhanced = torch.cat(outputs, dim=0)
                h_enhanced = self.final_layer_norm(h_enhanced + original_h_joint)
                h_final.index_copy_(0, head_indices, h_enhanced.expand(head_indices.size(0), -1))

        h_fake = self.calibrated_fusion(h, fake_hv, fake_ht, rg)
        t_fake = self.calibrated_fusion(t, fake_tv, fake_tt, rg)
        score_h = self.margin - self._calc(h_fake, t_final, r, mode)
        score_t = self.margin - self._calc(h_final, t_fake, r, mode)
        score_all = self.margin - self._calc(h_fake, t_fake, r, mode)
        return [score_h, score_t, score_all], [h_fake, r, t_fake]

    def forward_and_return_embs(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']

        
        batch_h = self._safe_ids(batch_h, self.ent_tot, "batch_h")
        batch_t = self._safe_ids(batch_t, self.ent_tot, "batch_t")
        batch_r = self._safe_ids(batch_r, self.rel_tot, "batch_r")

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        
        rg = self.rel_gate(batch_r)

        h_joint = self.calibrated_fusion(h, h_img_emb, h_text_emb, rg)
        t_joint = self.calibrated_fusion(t, t_img_emb, t_text_emb, rg)

        
        with torch.no_grad():
            S = min(1024, batch_h.size(0))
            batch_t_init = batch_t[:S]
            batch_h_init = batch_h[:S]
            h_init = h[:S]
            t_init = t[:S]
            h_joint_init = h_joint[:S]
            t_joint_init = t_joint[:S]

            num_heads = self.num_heads
            head_dim = self.head_dim
            temperature = self._enhancement_temp
            k_top = self._enhancement_topk

            # tails
            t_final = t_joint
            unique_tails = torch.unique(batch_t_init)
            for i in range(unique_tails.size(0)):
                tail_id = unique_tails[i]
                tail_indices = torch.where(batch_t_init == tail_id)[0]
                if tail_indices.numel() == 0:
                    continue

                related_h = h_init.index_select(0, tail_indices)
                related_h_joint = h_joint_init.index_select(0, tail_indices)

                first_idx = tail_indices[0]
                current_t = t_init[first_idx]
                original_t_joint = t_joint_init[first_idx]

                outputs = []
                amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if self._use_amp_enhancement else contextlib.nullcontext()
                with amp_ctx:
                    for head in range(num_heads):
                        s = head * head_dim; e = s + head_dim
                        q_in = current_t[s:e].unsqueeze(0)
                        k_in = related_h[:, s:e]
                        v_in = related_h_joint[:, s:e]

                        query = self.q_proj[head](q_in)
                        keys = self.k_proj[head](k_in)
                        scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(head_dim)
                        w, idx = self._topk_softmax(scores, min(k_top, k_in.size(0)), temperature)

                        values = self.v_proj[head](v_in)
                        sel_idx = idx.squeeze(0)
                        values_k = values.index_select(0, sel_idx)
                        cur_slice = original_t_joint[s:e].unsqueeze(0).expand(values_k.size(0), -1)
                        fused_k = self.negotiation_gate_proj(head, values_k, cur_slice)
                        head_out = torch.matmul(w, fused_k).squeeze(0)
                        outputs.append(head_out.float() if self._use_amp_enhancement else head_out)

                t_enhanced = torch.cat(outputs, dim=0)
                t_enhanced = self.final_layer_norm(t_enhanced + original_t_joint)
                t_final.index_copy_(0, tail_indices, t_enhanced.expand(tail_indices.size(0), -1))

            # heads
            h_final = h_joint
            unique_heads = torch.unique(batch_h_init)
            for i in range(unique_heads.size(0)):
                head_id = unique_heads[i]
                head_indices = torch.where(batch_h_init == head_id)[0]
                if head_indices.numel() == 0:
                    continue

                related_t = t_init.index_select(0, head_indices)
                related_t_joint = t_joint_init.index_select(0, head_indices)

                first_idx = head_indices[0]
                current_h = h_init[first_idx]
                original_h_joint = h_joint_init[first_idx]

                outputs = []
                amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if self._use_amp_enhancement else contextlib.nullcontext()
                with amp_ctx:
                    for head in range(num_heads):
                        s = head * head_dim; e = s + head_dim
                        q_in = current_h[s:e].unsqueeze(0)
                        k_in = related_t[:, s:e]
                        v_in = related_t_joint[:, s:e]

                        query = self.q_proj[head](q_in)
                        keys = self.k_proj[head](k_in)
                        scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(head_dim)
                        w, idx = self._topk_softmax(scores, min(k_top, k_in.size(0)), temperature)

                        values = self.v_proj[head](v_in)
                        sel_idx = idx.squeeze(0)
                        values_k = values.index_select(0, sel_idx)
                        cur_slice = original_h_joint[s:e].unsqueeze(0).expand(values_k.size(0), -1)
                        fused_k = self.negotiation_gate_proj(head, values_k, cur_slice)
                        head_out = torch.matmul(w, fused_k).squeeze(0)
                        outputs.append(head_out.float() if self._use_amp_enhancement else head_out)

                h_enhanced = torch.cat(outputs, dim=0)
                h_enhanced = self.final_layer_norm(h_enhanced + original_h_joint)
                h_final.index_copy_(0, head_indices, h_enhanced.expand(head_indices.size(0), -1))

        score = self.margin - self._calc(h_final, t_final, r, mode)
        return score, [h_final, r, t_final]

    def forward_test(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        original_t = data['original_t']
        original_h = data['original_h']
        t_final = data['t_final']
        h_final = data['h_final']

        batch_h = self._safe_ids(batch_h, self.ent_tot, "batch_h")
        batch_t = self._safe_ids(batch_t, self.ent_tot, "batch_t")
        batch_r = self._safe_ids(batch_r, self.rel_tot, "batch_r")

        h = self.ent_embeddings(batch_h)
        r = self.rel_embeddings(batch_r)

        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        
        rg = self.rel_gate(batch_r)

        h_joint = self.calibrated_fusion(h, h_img_emb, h_text_emb, rg)

        h_joint_updated = h_joint.clone()
        for i, h_id in enumerate(batch_h):
            mask = (original_h == h_id)
            if mask.any():
                index = torch.where(mask)[0][0]
                h_emb_mapped = h_final[index].unsqueeze(0)
                h_joint_updated[i] = h_emb_mapped

        t = batch_t[0]
        mask = (original_t == t)
        if mask.any():
            index = torch.where(mask)[0][0]
        else:
            raise ValueError(f"Tail entity {t.item()} not found in original_t.")
        t_emb_mapped = t_final[index].unsqueeze(0)

        score = self.margin - self._calc(h_joint_updated, t_emb_mapped, r, mode)
        return score

    def forward_test_tail(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        original_t = data['original_t']
        original_h = data['original_h']
        t_final = data['t_final']
        h_final = data['h_final']

        batch_h = self._safe_ids(batch_h, self.ent_tot, "batch_h")
        batch_t = self._safe_ids(batch_t, self.ent_tot, "batch_t")
        batch_r = self._safe_ids(batch_r, self.rel_tot, "batch_r")

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        
        rg = self.rel_gate(batch_r)

        h_joint = self.calibrated_fusion(h, h_img_emb, h_text_emb, rg)

        h_id = batch_h[0]
        mask = (original_h == h_id)
        if mask.any():
            index = torch.where(mask)[0][0]
        else:
            raise ValueError(f"HEAD entity {h_id.item()} not found in original_h.")
        h_emb_mapped = h_final[index].unsqueeze(0)

        t_joint = self.calibrated_fusion(t, t_img_emb, t_text_emb, rg)
        t_joint_updated = t_joint.clone()
        for i, t_id in enumerate(batch_t):
            mask = (original_t == t_id)
            if mask.any():
                index = torch.where(mask)[0][0]
                t_emb_mapped = t_final[index].unsqueeze(0)
                t_joint_updated[i] = t_emb_mapped
        score = self.margin - self._calc(h_emb_mapped, t_joint_updated, r, mode)
        return score

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()

    def predict_test(self, data):
        score = -self.forward_test(data)
        return score.cpu().data.numpy()

    def predict_test_tail(self, data):
        score = -self.forward_test_tail(data)
        return score.cpu().data.numpy()

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        batch_h = self._safe_ids(batch_h, self.ent_tot, "batch_h")
        batch_t = self._safe_ids(batch_t, self.ent_tot, "batch_t")
        batch_r = self._safe_ids(batch_r, self.rel_tot, "batch_r")

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul

    def attention_weight(self, es, ev, et, rg):
        e = torch.stack((es, ev, et), dim=1)
        u = torch.tanh(e)
        scores = self.V_global(u).squeeze(-1)
        attention_weights = torch.softmax(scores / torch.sigmoid(rg), dim=-1)
        return attention_weights

    def get_attention_weights(self, batch_h, batch_r, batch_t):
        batch_h = torch.LongTensor(batch_h)
        batch_r = torch.LongTensor(batch_r)
        batch_t = torch.LongTensor(batch_t)

        batch_h = self._safe_ids(batch_h, self.ent_tot, "batch_h")
        batch_t = self._safe_ids(batch_t, self.ent_tot, "batch_t")
        batch_r = self._safe_ids(batch_r, self.rel_tot, "batch_r")

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        
        rg = self.rel_gate(batch_r)

        weights_h = self.attention_weight(h, h_img_emb, h_text_emb, rg)
        weights_t = self.attention_weight(t, t_img_emb, t_text_emb, rg)
        return weights_h, weights_t

    def forward_and_return_embs_for_test(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']

        batch_h = self._safe_ids(batch_h, self.ent_tot, "batch_h")
        batch_t = self._safe_ids(batch_t, self.ent_tot, "batch_t")
        batch_r = self._safe_ids(batch_r, self.rel_tot, "batch_r")

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))

        rg = self.rel_gate(batch_r)

        h_joint = self.calibrated_fusion(h, h_img_emb, h_text_emb, rg)
        t_joint = self.calibrated_fusion(t, t_img_emb, t_text_emb, rg)

        batch_t_init = batch_t
        batch_h_init = batch_h
        h_init = h
        t_init = t
        h_joint_init = h_joint
        t_joint_init = t_joint

        with torch.no_grad():
            num_heads = self.num_heads
            head_dim = self.head_dim
            temperature = self._enhancement_temp
            k_top = self._enhancement_topk

            # tails
            t_final = t_joint
            unique_tails = torch.unique(batch_t_init)
            for i in range(unique_tails.size(0)):
                tail_id = unique_tails[i]
                tail_indices = torch.where(batch_t_init == tail_id)[0]
                if tail_indices.numel() == 0:
                    continue

                related_h = h_init.index_select(0, tail_indices)
                related_h_joint = h_joint_init.index_select(0, tail_indices)

                first_idx = tail_indices[0]
                current_t = t_init[first_idx]
                original_t_joint = t_joint_init[first_idx]

                t_final_head = []
                amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if self._use_amp_enhancement else contextlib.nullcontext()
                with amp_ctx:
                    for head in range(num_heads):
                        s = head * head_dim; e = s + head_dim
                        q_in = current_t[s:e].unsqueeze(0)
                        k_in = related_h[:, s:e]
                        v_in = related_h_joint[:, s:e]

                        query = self.q_proj[head](q_in)
                        keys = self.k_proj[head](k_in)
                        attention_scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(head_dim)
                        w, idx = self._topk_softmax(attention_scores, min(k_top, k_in.size(0)), temperature)

                        values = self.v_proj[head](v_in)
                        sel_idx = idx.squeeze(0)
                        values_k = values.index_select(0, sel_idx)
                        current_t_slice = original_t_joint[s:e].unsqueeze(0).expand(values_k.size(0), -1)
                        fused_k = self.negotiation_gate_proj(head, values_k, current_t_slice)
                        head_output = torch.matmul(w, fused_k)  # [1, Hd]
                        t_final_head.append(head_output.squeeze(0))

                t_enhanced = torch.cat(t_final_head, dim=0)
                t_enhanced = self.final_layer_norm(t_enhanced + original_t_joint)
                t_final.index_copy_(0, tail_indices, t_enhanced.expand(tail_indices.size(0), -1))

            # heads
            h_final = h_joint
            unique_heads = torch.unique(batch_h_init)
            for i in range(unique_heads.size(0)):
                head_id = unique_heads[i]
                head_indices = torch.where(batch_h_init == head_id)[0]
                if head_indices.numel() == 0:
                    continue

                related_t = t_init.index_select(0, head_indices)
                related_t_joint = t_joint_init.index_select(0, head_indices)

                first_idx = head_indices[0]
                current_h = h_init[first_idx]
                original_h_joint = h_joint_init[first_idx]

                h_final_head = []
                amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if self._use_amp_enhancement else contextlib.nullcontext()
                with amp_ctx:
                    for head in range(num_heads):
                        s = head * head_dim; e = s + head_dim
                        q_in = current_h[s:e].unsqueeze(0)
                        k_in = related_t[:, s:e]
                        v_in = related_t_joint[:, s:e]

                        query = self.q_proj[head](q_in)
                        keys = self.k_proj[head](k_in)
                        attention_scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(head_dim)
                        w, idx = self._topk_softmax(attention_scores, min(k_top, k_in.size(0)), temperature)

                        values = self.v_proj[head](v_in)
                        sel_idx = idx.squeeze(0)
                        values_k = values.index_select(0, sel_idx)
                        current_h_slice = original_h_joint[s:e].unsqueeze(0).expand(values_k.size(0), -1)
                        fused_k = self.negotiation_gate_proj(head, values_k, current_h_slice)
                        head_output = torch.matmul(w, fused_k)  # [1, Hd]
                        h_final_head.append(head_output.squeeze(0))

                h_enhanced = torch.cat(h_final_head, dim=0)
                h_enhanced = self.final_layer_norm(h_enhanced + original_h_joint)
                h_final.index_copy_(0, head_indices, h_enhanced.expand(head_indices.size(0), -1))

        return [h_final, r, t_final]
