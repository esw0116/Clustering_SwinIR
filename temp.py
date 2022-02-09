import torch

def construct_centroid(
        num_groups,
        points_flat: torch.Tensor,
        labels: torch.Tensor) -> torch.Tensor:

    # batch size
    b, n, n_feats = points_flat.size()
    # b = labels.size(0)

    # (B x k, C)
    points_flat = points_flat.view(b*n, n_feats)
    centroids_flat = points_flat.new_zeros(b * num_groups, n_feats)
    ones_flat = points_flat.new_ones(b*n).long()

    label_adder = num_groups * torch.arange(
        b,
        dtype=torch.long,
        device=labels.device,
    )
    label_adder = label_adder.view(-1, 1)

    # (B x N)
    labels_flat = (labels + label_adder).view(-1)
    centroids_flat.index_add_(0, labels_flat, points_flat)

    counts_flat = labels_flat.new_zeros(b * num_groups)
    counts_flat = counts_flat.index_add_(0, labels_flat, ones_flat)

    centroids_flat.div_(counts_flat.unsqueeze(-1) + 1e-8)
    centroids = centroids_flat.view(b, num_groups, n_feats)
    return centroids

b = 2
g = 8
n = 20

src = torch.randn((b, n, 5))
labels = torch.randint(g, (b, n))

label_ones = torch.ones_like(labels)
cnt_labels = torch.zeros(b, g).type_as(labels)
cnt_labels.scatter_add_(dim=1, index=labels, src=label_ones)

centers = construct_centroid(g, src, labels)
breakpoint()