import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score as sklearn_f1

from model import ProtoTFModel, ContrastiveLoss, TimeSeriesAugmentation
from utils import (
    set_seed, build_dataset_ucr, build_dataset_uea,
    TimeDatasetWithIndex, shuffler, flip_label, evaluate_model,
    adjust_learning_rate, SAM
)


try:
    from tsaug import TimeWarp as TsaugTimeWarp
    TSAUG_AVAILABLE = True
except ImportError:
    TSAUG_AVAILABLE = False


class TemporalAugmentationExpander:
    def __init__(self, n_speed_change=5, max_speed_ratio=3):
        if TSAUG_AVAILABLE:
            self.time_warp = TsaugTimeWarp(n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)
        else:
            self.time_warp = None

    def apply_time_warp(self, x):
        if not TSAUG_AVAILABLE or self.time_warp is None:
            return x
        device = x.device
        x_np = x.permute(0, 2, 1).cpu().numpy()
        warped_np = self.time_warp.augment(x_np)
        return torch.from_numpy(warped_np).float().permute(0, 2, 1).to(device)


class FrequencyDomainAugmenter:
    @staticmethod
    def amplitude_perturbation(x, sigma=0.1):
        x_fft = torch.fft.rfft(x, dim=-1)
        amplitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        noise = torch.randn_like(amplitude) * sigma
        perturbed_amplitude = torch.clamp(amplitude * (1 + noise), min=0)
        x_fft_perturbed = perturbed_amplitude * torch.exp(1j * phase)
        return torch.fft.irfft(x_fft_perturbed, n=x.shape[-1])

    @staticmethod
    def phase_shift(x, max_shift=0.2):
        x_fft = torch.fft.rfft(x, dim=-1)
        amplitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        shift = (torch.rand_like(phase) * 2 - 1) * max_shift * np.pi
        x_fft_shifted = amplitude * torch.exp(1j * (phase + shift))
        return torch.fft.irfft(x_fft_shifted, n=x.shape[-1])

    @staticmethod
    def frequency_masking(x, mask_ratio=0.1):
        x_fft = torch.fft.rfft(x, dim=-1)
        freq_len = x_fft.shape[-1]
        mask = torch.ones_like(x_fft, dtype=torch.float32)
        num_mask = int(freq_len * mask_ratio)
        for b in range(x.shape[0]):
            mask_indices = torch.randperm(freq_len)[:num_mask]
            mask[b, :, mask_indices] = 0
        x_fft_masked = x_fft * mask
        return torch.fft.irfft(x_fft_masked, n=x.shape[-1])

    def augment(self, x, strength='medium'):
        if strength == 'medium':
            if torch.rand(1).item() > 0.5:
                x = self.amplitude_perturbation(x, sigma=0.1)
            else:
                x = self.phase_shift(x, max_shift=0.15)
            if torch.rand(1).item() > 0.7:
                x = self.frequency_masking(x, mask_ratio=0.05)
        return x


class DataExpansionManager:
    def __init__(self, num_classes, device='cuda', n_speed_change=5, max_speed_ratio=3):
        self.num_classes = num_classes
        self.device = device
        self.temporal_expander = TemporalAugmentationExpander(n_speed_change, max_speed_ratio)
        self.frequency_augmenter = FrequencyDomainAugmenter()

    def expand_training_data(self, data, labels, sample_weights, num_temporal_aug=1, use_frequency_aug=True):
        device = data.device
        original_size = len(data)

        if isinstance(sample_weights, np.ndarray):
            sample_weights = torch.from_numpy(sample_weights).float().to(device)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long().to(device)

        all_data = [data]
        all_labels = [labels]
        all_weights = [sample_weights]

        selected_mask = sample_weights > 0
        selected_indices = torch.where(selected_mask)[0]

        if len(selected_indices) > 0 and num_temporal_aug > 0:
            selected_data = data[selected_indices]
            selected_labels = labels[selected_indices]
            selected_weights = sample_weights[selected_indices]

            for _ in range(num_temporal_aug):
                aug_data = self.temporal_expander.apply_time_warp(selected_data)
                if use_frequency_aug and torch.rand(1).item() > 0.5:
                    aug_data = self.frequency_augmenter.augment(aug_data, strength='medium')
                all_data.append(aug_data)
                all_labels.append(selected_labels.clone())
                all_weights.append(selected_weights.clone())

        expanded_data = torch.cat(all_data, dim=0)
        expanded_labels = torch.cat(all_labels, dim=0)
        expanded_weights = torch.cat(all_weights, dim=0)

        return expanded_data, expanded_labels, expanded_weights, {
            'original_size': original_size,
            'expanded_size': len(expanded_data),
            'expansion_ratio': len(expanded_data) / original_size,
            'num_selected': int(selected_mask.sum().item()),
        }


class SoftWeightSampleSelector:
    def __init__(self, num_samples, num_classes, k_neighbors=10, device='cuda'):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.k = k_neighbors
        self.device = device
        self.global_clean_mask = None
        self.global_clean_weights = None

    def compute_intra_class_similarity(self, features, labels):
        features = F.normalize(features, dim=1)
        n = features.shape[0]
        sim_matrix = torch.mm(features, features.t())
        intra_sim = torch.zeros(n, device=features.device)

        for c in range(self.num_classes):
            class_mask = labels == c
            class_indices = torch.where(class_mask)[0]
            if len(class_indices) <= 1:
                continue
            for idx in class_indices:
                other_same_class = class_mask.clone()
                other_same_class[idx] = False
                if other_same_class.sum() > 0:
                    intra_sim[idx] = sim_matrix[idx, other_same_class].mean()
        return intra_sim

    def compute_neighbor_agreement(self, features, labels):
        n = features.shape[0]
        k = min(self.k, n - 1)
        distances = torch.cdist(features, features)
        _, indices = torch.topk(distances, k + 1, largest=False, dim=1)
        neighbor_indices = indices[:, 1:]

        agreement_scores = torch.zeros(n, device=features.device)
        for i in range(n):
            neighbors = neighbor_indices[i]
            neighbor_labels = labels[neighbors]
            agreement = (neighbor_labels == labels[i]).float().mean()
            agreement_scores[i] = agreement
        return agreement_scores

    def global_normalize(self, scores):
        score_min = scores.min()
        score_max = scores.max()
        if score_max - score_min > 1e-6:
            return (scores - score_min) / (score_max - score_min)
        return torch.ones_like(scores) * 0.5

    def compute_trimmed_prototype_margin(self, freq_features, labels, trim_ratio=0.6):
        fn = F.normalize(freq_features, dim=1)
        N = fn.shape[0]
        num_classes = self.num_classes
        device = fn.device

        protos = []
        for c in range(num_classes):
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                protos.append(torch.zeros(fn.shape[1], device=device))
            else:
                p = fn[idx].mean(0)
                protos.append(F.normalize(p, dim=0))
        P = torch.stack(protos)

        sim_all = fn @ P.T
        assigned = sim_all[torch.arange(N), labels]
        best_other = torch.full((N,), -1.0, device=device)
        for i in range(N):
            others = [c for c in range(num_classes) if c != labels[i].item()]
            if others:
                best_other[i] = sim_all[i, others].max()
        prov_margin = assigned - best_other

        protos_trim = []
        for c in range(num_classes):
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                protos_trim.append(torch.zeros(fn.shape[1], device=device))
                continue
            if len(idx) >= 4:
                margins_c = prov_margin[idx]
                k = max(1, int(len(idx) * trim_ratio))
                top_k = margins_c.topk(k).indices
                kept = idx[top_k]
                p = fn[kept].mean(0)
            else:
                p = fn[idx].mean(0)
            protos_trim.append(F.normalize(p, dim=0))
        P_trim = torch.stack(protos_trim)

        sim_trim = fn @ P_trim.T
        assigned_trim = sim_trim[torch.arange(N), labels]
        best_other_trim = torch.full((N,), -1.0, device=device)
        for i in range(N):
            others = [c for c in range(num_classes) if c != labels[i].item()]
            if others:
                best_other_trim[i] = sim_trim[i, others].max()
        return assigned_trim - best_other_trim

    def compute_soft_weights(self, features, labels, freq_features=None, min_weight=0.5):
        intra_sim = self.compute_intra_class_similarity(features, labels)
        neighbor_score = self.compute_neighbor_agreement(features, labels)

        intra_sim_norm = self.global_normalize(intra_sim)
        neighbor_norm = self.global_normalize(neighbor_score)

        if freq_features is not None:
            m_trim = self.compute_trimmed_prototype_margin(freq_features, labels, trim_ratio=0.6)
            m_trim_norm = self.global_normalize(m_trim)
            combined_score = 0.35 * intra_sim_norm + 0.35 * neighbor_norm + 0.30 * m_trim_norm
        else:
            combined_score = 0.5 * intra_sim_norm + 0.5 * neighbor_norm

        sample_weights = torch.zeros_like(combined_score)

        for c in range(self.num_classes):
            class_mask = labels == c
            if class_mask.sum() <= 1:
                sample_weights[class_mask] = 1.0
                continue

            class_scores = combined_score[class_mask]
            class_mean = class_scores.mean()
            class_max = class_scores.max()

            weights = torch.zeros_like(class_scores)
            selected_mask = class_scores >= class_mean

            if selected_mask.sum() > 0:
                if class_max > class_mean:
                    selected_scores = class_scores[selected_mask]
                    ratio = (selected_scores - class_mean) / (class_max - class_mean + 1e-8)
                    weights[selected_mask] = min_weight + (1.0 - min_weight) * ratio
                else:
                    weights[selected_mask] = 1.0

            sample_weights[class_mask] = weights

        clean_mask = (sample_weights > 0).float()
        self.global_clean_mask = clean_mask.cpu().numpy()
        self.global_clean_weights = sample_weights.cpu().numpy()

        return self.global_clean_mask, self.global_clean_weights


def pretrain_ssl(model, train_loader, device, args):
    print("Phase 1: Self-Supervised Pretraining")
    model.train()
    contrastive_loss = ContrastiveLoss(temperature=args.ssl_temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.ssl_lr, weight_decay=1e-4)

    for epoch in range(args.ssl_epochs):
        total_loss = 0
        num_batches = 0

        for data, target, indices in train_loader:
            data = data.to(device)
            aug1 = TimeSeriesAugmentation.augment(data, strength='medium')
            aug2 = TimeSeriesAugmentation.augment(data, strength='medium')

            optimizer.zero_grad()

            temporal_feat1, frequency_feat1 = model.encoder(aug1, return_dual=True)
            temporal_feat2, frequency_feat2 = model.encoder(aug2, return_dual=True)

            loss = (contrastive_loss(temporal_feat1, frequency_feat1) +
                    contrastive_loss(temporal_feat2, frequency_feat2) +
                    contrastive_loss(temporal_feat1, temporal_feat2))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 20 == 0:
            print(f"  SSL Epoch {epoch+1}/{args.ssl_epochs}, Loss: {total_loss/num_batches:.4f}")

    print("  Pretraining completed.")
    return model


def extract_all_features(model, data_loader, device):
    model.eval()
    all_fused, all_freq, all_labels, all_indices = [], [], [], []

    with torch.no_grad():
        for data, target, indices in data_loader:
            data, target = data.to(device), target.to(device)
            _, fused_feat, _, freq_feat = model.forward_dual(data)
            all_fused.append(fused_feat)
            all_freq.append(freq_feat)
            all_labels.append(target)
            all_indices.append(indices)

    all_fused   = torch.cat(all_fused,   dim=0)
    all_freq    = torch.cat(all_freq,    dim=0)
    all_labels  = torch.cat(all_labels,  dim=0)
    all_indices = torch.cat(all_indices, dim=0)

    sort_order = all_indices.argsort()
    model.train()
    return all_fused[sort_order], all_freq[sort_order], all_labels[sort_order]


def print_selection_metrics(pred_clean_mask, true_noise_mask, epoch):
    pred_clean = pred_clean_mask.astype(int)
    true_clean = (1 - true_noise_mask).astype(int)
    precision = precision_score(true_clean, pred_clean, zero_division=0)
    recall    = recall_score(true_clean, pred_clean, zero_division=0)
    f1        = sklearn_f1(true_clean, pred_clean, zero_division=0)
    print(f"  [Selection] Epoch {epoch}: "
          f"Pred: {pred_clean.sum()}/{len(pred_clean)}, "
          f"P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")


def main_train_with_expansion(model, train_loader, test_loader, train_data, noisy_labels,
                               sample_selector, noise_mask, device, args):
    print("Phase 2: Main Training")
    model.train()

    data_expander = DataExpansionManager(
        args.num_classes, device,
        n_speed_change=args.n_speed_change,
        max_speed_ratio=args.max_speed_ratio
    )

    optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=args.sam_rho,
                    lr=args.lr, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss(reduction='none')

    best_acc = 0
    test_accuracies = []

    train_data_tensor   = torch.from_numpy(train_data).float().to(device)
    noisy_labels_tensor = torch.from_numpy(noisy_labels).long().to(device)

    expanded_data    = None
    expanded_weights = None

    for epoch in range(args.epochs):
        model.train()
        adjust_learning_rate(optimizer.base_optimizer, epoch, args.lr, args.epochs, warmup_epochs=10)

        all_features, all_freq_features, all_labels = extract_all_features(model, train_loader, device)
        global_clean_mask, global_sample_weights = sample_selector.compute_soft_weights(
            all_features, all_labels, freq_features=all_freq_features, min_weight=args.min_weight
        )
        print_selection_metrics(global_clean_mask, noise_mask, epoch)

        if epoch % 5 == 0 and epoch >= 10:
            expanded_data, expanded_labels, expanded_weights, exp_info = \
                data_expander.expand_training_data(
                    train_data_tensor, noisy_labels_tensor,
                    global_sample_weights, num_temporal_aug=args.num_temporal_aug
                )
            print(f"  [Expansion] {exp_info['original_size']} -> {exp_info['expanded_size']} "
                  f"({exp_info['expansion_ratio']:.2f}x)")

            expanded_dataset = TensorDataset(
                expanded_data, expanded_labels,
                torch.arange(len(expanded_data), device=device)
            )
            expanded_loader = DataLoader(expanded_dataset, batch_size=args.batch_size, shuffle=True)
        else:
            if expanded_data is None:
                expanded_loader = train_loader
                expanded_weights = torch.from_numpy(global_sample_weights).float().to(device)

        current_loader = expanded_loader if expanded_data is not None else train_loader
        total_loss = 0
        num_batches = 0

        for batch_data in current_loader:
            if len(batch_data) == 3:
                data, target, indices = batch_data
            else:
                data, target = batch_data[:2]
                indices = torch.arange(len(data))

            data, target = data.to(device), target.to(device)
            indices_np = indices.cpu().numpy()

            if expanded_weights is not None and indices_np.max() < len(expanded_weights):
                batch_weights = expanded_weights[indices_np]
                if isinstance(batch_weights, np.ndarray):
                    batch_weights = torch.from_numpy(batch_weights).float().to(device)
            else:
                batch_weights = torch.ones(len(data), device=device)

            logits, _, _, _ = model.forward_dual(data)
            sample_losses = ce_loss(logits, target)
            valid_mask = batch_weights > 0
            if valid_mask.sum() > 0:
                loss = (sample_losses * batch_weights * valid_mask).sum() / (batch_weights[valid_mask].sum() + 1e-10)
            else:
                loss = sample_losses.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step(zero_grad=True)

            logits2, _, _, _ = model.forward_dual(data)
            loss2 = ce_loss(logits2, target)
            if valid_mask.sum() > 0:
                loss2 = (loss2 * batch_weights * valid_mask).sum() / (batch_weights[valid_mask].sum() + 1e-10)
            else:
                loss2 = loss2.mean()
            loss2.backward()
            optimizer.second_step(zero_grad=True)

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0 or epoch >= args.epochs - 5:
            test_acc, test_f1 = evaluate_model(model, test_loader, device)
            test_accuracies.append(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/num_batches:.4f}, "
                  f"Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

    final_acc = np.mean(test_accuracies[-5:]) if len(test_accuracies) >= 5 else test_accuracies[-1]
    return final_acc, best_acc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--cuda', type=str, default='cuda:0')

    parser.add_argument('--dataset', type=str, default='BasicMotions')
    parser.add_argument('--archive', type=str, default='UEA', choices=['UCR', 'UEA'])
    parser.add_argument('--data_path', type=str, default='../data/Multivariate2018_arff',
                        help='Root directory containing dataset folders')
    parser.add_argument('--num_classes', type=int, default=0)
    parser.add_argument('--input_size', type=int, default=1)

    parser.add_argument('--label_noise_type', type=int, default=0)
    parser.add_argument('--label_noise_rate', type=float, default=0.3)

    parser.add_argument('--hidden_dim', type=int, default=128)

    parser.add_argument('--ssl_epochs', type=int, default=100)
    parser.add_argument('--ssl_lr', type=float, default=0.001)
    parser.add_argument('--ssl_temperature', type=float, default=0.5)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sam_rho', type=float, default=0.05)

    parser.add_argument('--k_neighbors', type=int, default=10)
    parser.add_argument('--min_weight', type=float, default=0.5)
    parser.add_argument('--num_temporal_aug', type=int, default=1)
    parser.add_argument('--n_speed_change', type=int, default=5)
    parser.add_argument('--max_speed_ratio', type=float, default=3.0)

    args = parser.parse_args()

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_seed(args)

    print(f"Loading dataset: {args.dataset} ({args.archive})")
    if args.archive == 'UCR':
        train_data, train_labels, test_data, test_labels, num_classes = build_dataset_ucr(args)
    else:
        train_data, train_labels, test_data, test_labels, num_classes = build_dataset_uea(args)

    args.num_classes = num_classes
    args.input_size  = train_data.shape[1]
    print(f"Train: {train_data.shape}, Test: {test_data.shape}, Classes: {num_classes}")

    train_data, train_labels = shuffler(train_data, train_labels)

    if args.label_noise_rate > 0:
        print(f"Adding noise: type={args.label_noise_type}, rate={args.label_noise_rate}")
        noisy_labels, noise_mask = flip_label(
            dataset=train_data, target=train_labels,
            ratio=args.label_noise_rate, args=args,
            pattern=args.label_noise_type
        )
        print(f"Actual noise rate: {noise_mask.mean():.4f}")
    else:
        noisy_labels = train_labels.copy()
        noise_mask   = np.zeros(len(train_labels))

    train_dataset = TimeDatasetWithIndex(
        torch.from_numpy(train_data).float().to(device),
        torch.from_numpy(noisy_labels).long().to(device)
    )
    test_dataset = TimeDatasetWithIndex(
        torch.from_numpy(test_data).float().to(device),
        torch.from_numpy(test_labels).long().to(device)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    model = ProtoTFModel(
        num_classes=num_classes,
        input_size=args.input_size,
        hidden_dim=args.hidden_dim
    ).to(device)

    sample_selector = SoftWeightSampleSelector(
        num_samples=len(train_data),
        num_classes=num_classes,
        k_neighbors=args.k_neighbors,
        device=device
    )

    start_time = time.time()

    if args.ssl_epochs > 0:
        model = pretrain_ssl(model, train_loader, device, args)

    final_acc, best_acc = main_train_with_expansion(
        model, train_loader, test_loader, train_data, noisy_labels,
        sample_selector, noise_mask, device, args
    )

    print(f"\nDataset: {args.dataset} | Noise type={args.label_noise_type}, rate={args.label_noise_rate}")
    print(f"Final Acc: {final_acc:.4f} | Best Acc: {best_acc:.4f}")
    print(f"Time: {time.time() - start_time:.2f}s")

    return final_acc, best_acc


if __name__ == '__main__':
    main()
