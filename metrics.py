
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.inference_mode()
def extract_features(model, loader, device, description):
    model.eval()
    features_list = []
    pids_all = []
    camids_all = []
    start = time.time()

    for batch_index, (images, pids, camids) in enumerate(loader, 1):
        images = images.to(device)
        features = model.forward_embeddings(images)
        features = F.normalize(features, p=2, dim=1)

        features_list.append(features.cpu())
        pids_all.extend(pids.tolist())
        camids_all.extend(camids.tolist())

        if batch_index % 50 == 0:
            print(f"{description}: batch {batch_index}/{len(loader)}")

    print(
        f"{description} feature extraction: "
        f"{time.time() - start:.1f}s"
    )

    return torch.cat(features_list, dim=0), pids_all, camids_all


@torch.inference_mode()
def evaluate_reid(model, query_loader, gallery_loader, device):
    query_features, query_pids, query_camids = extract_features(
        model, query_loader, device, "Query"
    )
    gallery_features, gallery_pids, gallery_camids = extract_features(
        model, gallery_loader, device, "Gallery"
    )

    distances = 1.0 - query_features @ gallery_features.T
    gallery_pids = torch.tensor(gallery_pids)
    gallery_camids = torch.tensor(gallery_camids)

    cmc_sum = torch.zeros(20, dtype=torch.float64)
    average_precisions = []
    valid_queries = 0

    for index, (query_pid, query_camid) in enumerate(
        zip(query_pids, query_camids)
    ):
        if query_pid <= 0:
            continue

        order = torch.argsort(distances[index])
        ordered_pids = gallery_pids[order]
        ordered_camids = gallery_camids[order]

        keep = (ordered_pids != -1) & ~(
            (ordered_pids == query_pid) &
            (ordered_camids == query_camid)
        )

        matches = (ordered_pids[keep] == query_pid)

        if not matches.any():
            continue

        valid_queries += 1
        match_positions = torch.where(matches)[0]

        first_match = int(match_positions[0])
        end = min(20, len(matches))

        if first_match < end:
            cmc_sum[first_match:end] += 1

        ranks = torch.arange(
            1, len(matches) + 1, dtype=torch.float32
        )
        cumulative_matches = matches.to(torch.float32).cumsum(dim=0)

        precisions = (
            cumulative_matches[match_positions]
            / ranks[match_positions]
        )

        average_precisions.append(precisions.mean().item())

    if valid_queries == 0:
        raise RuntimeError(
            "No valid query matches found. Check Market-1501 files."
        )

    cmc = cmc_sum / valid_queries
    mean_ap = sum(average_precisions) / len(average_precisions)

    results = {
        "Rank-1": 100.0 * cmc[0].item(),
        "Rank-5": 100.0 * cmc[4].item(),
        "Rank-10": 100.0 * cmc[9].item(),
        "Rank-20": 100.0 * cmc[19].item(),
        "mAP": 100.0 * mean_ap,
        "Valid Queries": valid_queries,
    }

    print(f"Valid queries: {valid_queries}")
    print(f"Minimum AP: {min(average_precisions):.6f}")
    print(f"Maximum AP: {max(average_precisions):.6f}")
    print(f"Mean AP: {mean_ap:.6f}")
    print(f"Rank-1: {results['Rank-1']:.6f}%")
    print(f"Rank-5: {results['Rank-5']:.6f}%")
    print(f"Rank-10: {results['Rank-10']:.6f}%")
    print(f"Rank-20: {results['Rank-20']:.6f}%")
    print(f"mAP: {results['mAP']:.6f}%")

    return results


def model_complexity(model, input_shape, device):
    macs = 0
    handles = []

    def conv_hook(module, inputs, output):
        nonlocal macs
        if not isinstance(output, torch.Tensor):
            return

        kernel_h, kernel_w = module.kernel_size
        output_elements = output.numel()

        macs += (
            output_elements
            * (module.in_channels // module.groups)
            * kernel_h
            * kernel_w
        )

    def linear_hook(module, inputs, output):
        nonlocal macs
        if isinstance(output, torch.Tensor):
            macs += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(
                module.register_forward_hook(conv_hook)
            )
        elif isinstance(module, nn.Linear):
            handles.append(
                module.register_forward_hook(linear_hook)
            )

    was_training = model.training
    model.eval()

    try:
        dummy = torch.zeros(input_shape, device=device)

        with torch.inference_mode():
            model.forward_embeddings(dummy)
    finally:
        for handle in handles:
            handle.remove()

        model.train(was_training)

    inference_parameters = (
        sum(p.numel() for p in model.backbone.parameters())
        + sum(p.numel() for p in model.bottleneck.parameters())
    )

    total_parameters = sum(p.numel() for p in model.parameters())

    return {
        "Inference Parameters (M)": inference_parameters / 1e6,
        "Total Parameters (M)": total_parameters / 1e6,
        "MACs (G)": macs / 1e9,
        "FLOPs (G)": (2 * macs) / 1e9,
    }
