# Author: Yao
# CreatTime: 2025/10/17
# FileName: t_sne
# Description: Simple implementation of the t_sne
from sklearn.preprocessing import StandardScaler
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_tsne(features, labels, text_topk_indices, class_name = None, t_sne = None, mask_id=None):
    """Visualize clustering results with t-SNE."""
    if mask_id is not None:
        SPECIAL_LABELS = mask_id

    else:
        SPECIAL_LABELS = text_topk_indices

    SPECIAL_COLORS = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'purple'
    }
    # labels to highlight and their colors


    # 1. Preprocess data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 2. t-SNE dimensionality reduction
    start_time = time.time()

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        random_state=42,
        method='barnes_hut',
        angle=0.5,
        n_jobs=-1
    )

    tsne_emb = tsne.fit_transform(scaled_features)
    print(f"t-SNE done in {time.time() - start_time:.2f}s")

    # 3. Create figure
    plt.figure(figsize=(14, 12))

    # 4. Scatter plot
    unique_labels = np.unique(labels)

    # regular labels first (gray)
    for i, label in enumerate(unique_labels):
        #     continue
        if label in SPECIAL_LABELS:
            continue  # skip noise and highlighted labels

        mask = (labels == label)
        plt.scatter(
            tsne_emb[mask, 0], tsne_emb[mask, 1],
            color='gray',
            alpha=0.4,
            s=30,
            edgecolor='k',
            linewidth=0.3
        )
    # highlighted labels on top
    for i, label in enumerate(SPECIAL_LABELS):

        if t_sne is not None or mask_id is not None:
            mask = (labels == label)
        else:
            if len(label)==1:
                mask = (labels == label)
            elif len(label)==2:
                label1 = label[0]
                label2 = label[1]
                mask = (labels == label1) | (labels == label2)
            else:
                raise ValueError(f"Only 1-2 highlight labels supported, got {len(label)}.")
        if mask_id is not None:
            plt.scatter(
                tsne_emb[mask, 0], tsne_emb[mask, 1],
                color=SPECIAL_COLORS[int(text_topk_indices[i])],
                # label=f'{class_name[int(text_topk_indices[i])]}',
                alpha=0.9,
                s=40,
                edgecolor='k',
                linewidth=0.3
            )
        else:
            plt.scatter(
                tsne_emb[mask, 0], tsne_emb[mask, 1],
                color=SPECIAL_COLORS[i],
                label=f'{class_name[i]}',
                alpha=0.9,
                s=40,
                edgecolor='k',
                linewidth=0.3
            )

    #     plt.scatter(

    # 6. cluster-center markers for highlighted labels
    for i, label in enumerate(SPECIAL_LABELS):

        if t_sne is not None or mask_id is not None:
            mask = (labels == label)
        else:
            if len(label)==1:
                mask = (labels == label)
            elif len(label)==2:
                label1 = label[0]
                label2 = label[1]
                mask = (labels == label1) | (labels == label2)
        center = np.mean(tsne_emb[mask], axis=0) if tsne_emb[mask].size else None
        if center is not None and mask_id is not None:
            plt.scatter(
                center[0], center[1],
                color=SPECIAL_COLORS[int(text_topk_indices[i])],
                marker='*',
                s=400,
                edgecolor='gold',
                linewidth=1.5,
                # label=f'center'
            )
        elif mask_id is None:
            plt.scatter(
                center[0], center[1],
                color=SPECIAL_COLORS[i],
                marker='*',
                s=400,
                edgecolor='gold',
                linewidth=1.5,
                # label=f'center'
            )
        else:
            continue

    # 7. Polish figure
    plt.title('Cluster Visualization (t-SNE projection)', fontsize=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=20)
    plt.ylabel('t-SNE Dimension 2', fontsize=20)
    # axis limits
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    # 8. Legend
    plt.legend(
        markerscale=2,
        loc='upper right',
        framealpha=0.9,
        fontsize=30,
        ncol=1
    )

    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.show()