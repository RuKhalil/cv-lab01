import torch
import math
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 任务1：正弦位置编码 ======================
def sinusoidal_pe(seq_len: int, d_model: int):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe

# ====================== 任务2：2D向量旋转 ======================
def rotate_2d(x: torch.Tensor, theta: float):
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x1, x2 = x[0], x[1]
    x1_new = x1 * cos_t - x2 * sin_t
    x2_new = x1 * sin_t + x2 * cos_t
    return torch.tensor([x1_new, x2_new])

# ====================== 任务3：高维RoPE ======================
def rope_rotate(x: torch.Tensor, pos: torch.Tensor):
    batch, seq_len, d = x.shape
    assert d % 2 == 0
    theta = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pos = pos.unsqueeze(1)
    angle = pos * theta
    x_res = torch.zeros_like(x)
    for i in range(d // 2):
        x0 = x[:, :, 2*i]
        x1 = x[:, :, 2*i+1]
        cos_a = torch.cos(angle[:, i])
        sin_a = torch.sin(angle[:, i])
        x_res[:, :, 2*i] = x0 * cos_a - x1 * sin_a
        x_res[:, :, 2*i] = x0 * sin_a + x1 * cos_a
    return x_res

# ====================== 绘图部分（生成四张图） ======================
if __name__ == "__main__":
    batch_size = 1
    seq_len = 32
    d_model = 16

    # 1. 图1 pe_heatmap.png 正弦PE热力图
    pe_table = sinusoidal_pe(seq_len, d_model)
    plt.figure(figsize=(8,6))
    plt.imshow(pe_table.numpy(), cmap="RdBu")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.title("Sinusoidal Position Encoding Heatmap")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("pe_heatmap.png", dpi=150)
    plt.close()
    print("已生成：pe_heatmap.png")

    # 2. 图2 rope_feature.png 不同位置RoPE变换对比
    test_vec = torch.tensor([[[0.8, 0.2, 0.3, 0.7, 0.1, 0.9, 0.5, 0.4]]])
    pos0 = torch.tensor([0])
    pos10 = torch.tensor([10])
    vec_pos0 = rope_rotate(test_vec, pos0)[0,0].numpy()
    vec_pos10 = rope_rotate(test_vec, pos10)[0,0].numpy()

    fig, axes = plt.subplots(2,1, figsize=(7,5))
    axes[0].bar(np.arange(d_model//2), vec_pos0)
    axes[0].set_title("RoPE Feature (pos=0)")
    axes[1].bar(np.arange(d_model//2), vec_pos10)
    axes[1].set_title("RoPE Feature (pos=10)")
    plt.tight_layout()
    plt.savefig("rope_feature.png", dpi=150)
    plt.close()
    print("已生成：rope_feature.png")

    # 3. 图3 add_vs_rope.png 两种位置注入对比示意图
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    ax1.text(0.2,0.5,"Embedding E\n+\nPE Vector", fontsize=14, ha="center")
    ax1.set_title("E+PE Add Method")
    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1)
    ax1.axis("off")

    ax2.text(0.2,0.6,"Embedding E", fontsize=14, ha="center")
    ax2.text(0.7,0.6,"Q/K RoPE Rotate", fontsize=14, ha="center")
    ax2.annotate("", xy=(0.6,0.6), xytext=(0.35,0.6), arrowprops=dict(arrowstyle="->"))
    ax2.set_title("RoPE Method")
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig("add_vs_rope.png", dpi=150)
    plt.close()
    print("已生成：add_vs_rope.png")

    # 4. 图4 rope_relative.png 验证相对位置点积曲线
    def calc_dot(q, k, pos_q, pos_k):
        qr = rope_rotate(q, torch.tensor([pos_q]))
        kr = rope_rotate(k, torch.tensor([pos_k]))
        return torch.sum(qr*kr).item()

    q_test = torch.tensor([[[0.2,0.5,0.1,0.6,0.3,0.4,0.7,0.2]]])
    k_test = torch.tensor([[[0.4,0.1,0.5,0.2,0.6,0.3,0.1,0.8]]])
    rel_diffs = list(range(0,15))
    dot_vals = []
    for diff in rel_diffs:
        dot = calc_dot(q_test, k_test, pos_q=0, pos_k=diff)
        dot_vals.append(dot)

    plt.figure(figsize=(7,4))
    plt.plot(rel_diffs, dot_vals, marker="o")
    plt.xlabel("Relative Position Offset (n-m)")
    plt.ylabel("Q·K Dot Product")
    plt.title("RoPE Dot Product vs Relative Position")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("rope_relative.png", dpi=150)
    plt.close()
    print("已生成：rope_relative.png")

    # ---------------- 任务4 流程打印对比 ----------------
    batch_size = 1
    seq_len = 10
    d_model = 8
    embed = torch.nn.Embedding(100, d_model)
    tokens = torch.randint(0,100,(batch_size,seq_len))
    E = embed(tokens)
    pe = sinusoidal_pe(seq_len, d_model).unsqueeze(0)
    X_add = E + pe
    print("\n==== E+PE加法模式 ====")
    print(f"混合后特征shape: {X_add.shape}")

    pos_ids = torch.arange(seq_len)
    wq = torch.nn.Linear(d_model, d_model)
    wk = torch.nn.Linear(d_model, d_model)
    Q_raw = wq(E)
    K_raw = wk(E)
    Q_rot = rope_rotate(Q_raw, pos_ids)
    K_rot = rope_rotate(K_raw, pos_ids)
    print("\n==== RoPE旋转模式 ====")
    print(f"原始Embedding不修改，仅Q/K旋转")

    # ---------------- 任务5 数值验证打印 ----------------
    print("\n==== RoPE相对位置数值验证 ====")
    d1 = calc_dot(q_test, k_test, 2,5)
    d2 = calc_dot(q_test, k_test,4,7)
    d3 = calc_dot(q_test, k_test,10,13)
    print(f"相对差3 | (2,5)点积: {d1:.4f}")
    print(f"相对差3 | (4,7)点积: {d2:.4f}")
    print(f"相对差3 | (10,13)点积: {d3:.4f}")