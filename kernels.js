window.KERNELS = [
  {
    "id": "vector_add",
    "label": "Vector add",
    "source": "import torch, triton, triton.language as tl\n\n@triton.jit\ndef add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):\n    pid = tl.program_id(0)\n    off = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = off < n\n    x = tl.load(x_ptr + off, mask=mask)\n    y = tl.load(y_ptr + off, mask=mask)\n    tl.store(out_ptr + off, x + y, mask=mask)\n\ndef run():\n    n = 1024\n    x = torch.randn(n, device=\"mps\")\n    y = torch.randn(n, device=\"mps\")\n    out = torch.empty_like(x)\n    add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)\n    torch.mps.synchronize()\n    return (out - (x + y)).abs().max().item()\n",
    "cat": "triton"
  },
  {
    "id": "fused_relu",
    "label": "Fused ReLU x scale",
    "source": "import torch, triton, triton.language as tl\n\n@triton.jit\ndef relu_mul_kernel(x_ptr, s_ptr, out_ptr, n, BLOCK: tl.constexpr):\n    pid = tl.program_id(0)\n    off = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = off < n\n    x = tl.load(x_ptr + off, mask=mask)\n    s = tl.load(s_ptr + off, mask=mask)\n    y = tl.where(x > 0, x, 0.0) * s\n    tl.store(out_ptr + off, y, mask=mask)\n\ndef run():\n    n = 4096\n    x = torch.randn(n, device=\"mps\")\n    s = torch.randn(n, device=\"mps\")\n    out = torch.empty_like(x)\n    relu_mul_kernel[(triton.cdiv(n, 256),)](x, s, out, n, BLOCK=256)\n    torch.mps.synchronize()\n    ref = torch.relu(x) * s\n    return (out - ref).abs().max().item()\n",
    "cat": "triton"
  },
  {
    "id": "softmax",
    "label": "Softmax (row)",
    "source": "import torch, triton, triton.language as tl\n\n@triton.jit\ndef softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):\n    row = tl.program_id(0)\n    cols = tl.arange(0, BLOCK)\n    mask = cols < n_cols\n    ptr = x_ptr + row * n_cols + cols\n    x = tl.load(ptr, mask=mask, other=-float(\"inf\"))\n    x = x - tl.max(x, axis=0)\n    num = tl.exp(x)\n    den = tl.sum(num, axis=0)\n    tl.store(out_ptr + row * n_cols + cols, num / den, mask=mask)\n\ndef run():\n    rows, cols = 128, 512\n    x = torch.randn(rows, cols, device=\"mps\")\n    out = torch.empty_like(x)\n    BLOCK = triton.next_power_of_2(cols)\n    softmax_kernel[(rows,)](x, out, cols, BLOCK=BLOCK)\n    torch.mps.synchronize()\n    ref = torch.softmax(x, dim=1)\n    return (out - ref).abs().max().item()\n",
    "cat": "triton"
  },
  {
    "id": "conv1d",
    "label": "1D convolution",
    "source": "import torch, triton, triton.language as tl\n\n@triton.jit\ndef conv1d_kernel(x_ptr, w_ptr, out_ptr, n, K: tl.constexpr, BLOCK: tl.constexpr):\n    pid = tl.program_id(0)\n    off = pid * BLOCK + tl.arange(0, BLOCK)\n    acc = tl.zeros((BLOCK,), dtype=tl.float32)\n    for k in tl.static_range(K):\n        idx = off + k\n        x = tl.load(x_ptr + idx, mask=idx < n + K - 1, other=0.0)\n        acc += x * tl.load(w_ptr + k)\n    tl.store(out_ptr + off, acc, mask=off < n)\n\ndef run():\n    n, K = 4096, 5\n    x = torch.randn(n + K - 1, device=\"mps\")\n    w = torch.randn(K, device=\"mps\")\n    out = torch.empty(n, device=\"mps\")\n    conv1d_kernel[(triton.cdiv(n, 256),)](x, w, out, n, K=K, BLOCK=256)\n    torch.mps.synchronize()\n    ref = torch.stack([(x[i:i + K] * w).sum() for i in range(n)])\n    return (out - ref).abs().max().item()\n",
    "cat": "triton"
  },
  {
    "id": "layernorm",
    "label": "LayerNorm (row)",
    "source": "import torch, triton, triton.language as tl\n\n@triton.jit\ndef layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):\n    row = tl.program_id(0)\n    cols = tl.arange(0, BLOCK)\n    mask = cols < n_cols\n    ptr = x_ptr + row * n_cols + cols\n    x = tl.load(ptr, mask=mask, other=0.0)\n    mean = tl.sum(x, axis=0) / n_cols\n    xc = tl.where(mask, x - mean, 0.0)\n    var = tl.sum(xc * xc, axis=0) / n_cols\n    xhat = xc / tl.sqrt(var + eps)\n    w = tl.load(w_ptr + cols, mask=mask)\n    b = tl.load(b_ptr + cols, mask=mask)\n    tl.store(out_ptr + row * n_cols + cols, xhat * w + b, mask=mask)\n\ndef run():\n    rows, cols = 256, 768\n    x = torch.randn(rows, cols, device=\"mps\")\n    w = torch.randn(cols, device=\"mps\")\n    b = torch.randn(cols, device=\"mps\")\n    out = torch.empty_like(x)\n    BLOCK = triton.next_power_of_2(cols)\n    layernorm_kernel[(rows,)](x, w, b, out, cols, 1e-5, BLOCK=BLOCK)\n    torch.mps.synchronize()\n    ref = torch.nn.functional.layer_norm(x, (cols,), w, b, 1e-5)\n    return (out - ref).abs().max().item()\n",
    "cat": "triton"
  },
  {
    "id": "g_vecadd",
    "label": "Vector add",
    "cat": "gluon",
    "source": "import torch, triton\nfrom triton.experimental import gluon\nfrom triton.experimental.gluon import language as ttgl\n\nTHREADS_PER_WARP = ttgl.constexpr(32)\n\n@gluon.jit\ndef add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: ttgl.constexpr):\n    layout: ttgl.constexpr = ttgl.BlockedLayout([1], [THREADS_PER_WARP], [4], [0])\n    off = ttgl.program_id(0) * BLOCK + ttgl.arange(0, BLOCK, layout=layout)\n    mask = off < n\n    x = ttgl.load(x_ptr + off, mask)\n    y = ttgl.load(y_ptr + off, mask)\n    ttgl.store(out_ptr + off, x + y, mask)\n\ndef run():\n    n = 1024\n    x = torch.randn(n, device=\"mps\")\n    y = torch.randn(n, device=\"mps\")\n    out = torch.empty_like(x)\n    add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)\n    torch.mps.synchronize()\n    return (out - (x + y)).abs().max().item()\n"
  },
  {
    "id": "g_softmax",
    "label": "Softmax (row)",
    "cat": "gluon",
    "source": "import torch, triton\nfrom triton.experimental import gluon\nfrom triton.experimental.gluon import language as ttgl\n\nTHREADS_PER_WARP = ttgl.constexpr(32)\n\n@gluon.jit\ndef softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: ttgl.constexpr):\n    layout: ttgl.constexpr = ttgl.BlockedLayout([1], [THREADS_PER_WARP], [4], [0])\n    row = ttgl.program_id(0)\n    cols = ttgl.arange(0, BLOCK, layout=layout)\n    mask = cols < n_cols\n    ptr = x_ptr + row * n_cols + cols\n    x = ttgl.load(ptr, mask, other=-float(\"inf\"))\n    x = x - ttgl.max(x, axis=0)\n    num = ttgl.exp(x)\n    den = ttgl.sum(num, axis=0)\n    ttgl.store(out_ptr + row * n_cols + cols, num / den, mask)\n\ndef run():\n    rows, cols = 128, 512\n    x = torch.randn(rows, cols, device=\"mps\")\n    out = torch.empty_like(x)\n    BLOCK = triton.next_power_of_2(cols)\n    softmax_kernel[(rows,)](x, out, cols, BLOCK=BLOCK)\n    torch.mps.synchronize()\n    return (out - torch.softmax(x, dim=1)).abs().max().item()\n"
  },
  {
    "id": "g_matmul",
    "label": "Matmul (dot_fma)",
    "cat": "gluon",
    "source": "import torch, triton\nfrom triton.experimental import gluon\nfrom triton.experimental.gluon import language as ttgl\n\nTHREADS_PER_WARP = ttgl.constexpr(32)\n\n@gluon.jit\ndef matmul_kernel(a_ptr, b_ptr, c_ptr, out_ptr, B: ttgl.constexpr):\n    layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [THREADS_PER_WARP, 1], [ttgl.num_warps(), 1], [1, 0])\n    lhs: ttgl.constexpr = ttgl.DotOperandLayout(parent=layout, operand_index=0, k_width=0)\n    rhs: ttgl.constexpr = ttgl.DotOperandLayout(parent=layout, operand_index=1, k_width=0)\n    m = ttgl.arange(0, B, layout=ttgl.SliceLayout(1, layout))[:, None]\n    n = ttgl.arange(0, B, layout=ttgl.SliceLayout(0, layout))[None, :]\n    offs = m * B + n\n    a = ttgl.convert_layout(ttgl.load(a_ptr + offs), lhs)\n    b = ttgl.convert_layout(ttgl.load(b_ptr + offs), rhs)\n    c = ttgl.load(c_ptr + offs)\n    ttgl.store(out_ptr + offs, ttgl.dot_fma(a, b, c))\n\ndef run():\n    B = 32\n    a = torch.rand((B, B), device=\"mps\")\n    b = torch.rand((B, B), device=\"mps\")\n    c = torch.rand((B, B), device=\"mps\")\n    out = torch.empty((B, B), device=\"mps\")\n    matmul_kernel[(1,)](a, b, c, out, B=B)\n    torch.mps.synchronize()\n    return (out - torch.addmm(c, a, b)).abs().max().item()\n"
  }
];
