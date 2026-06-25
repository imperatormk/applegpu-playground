window.KERNELS = [
  {
    "id": "vector_add",
    "label": "Vector add",
    "source": "import torch, triton, triton.language as tl\n\n@triton.jit\ndef add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):\n    pid = tl.program_id(0)\n    off = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = off < n\n    x = tl.load(x_ptr + off, mask=mask)\n    y = tl.load(y_ptr + off, mask=mask)\n    tl.store(out_ptr + off, x + y, mask=mask)\n\ndef run():\n    n = 1024\n    x = torch.randn(n, device=\"mps\")\n    y = torch.randn(n, device=\"mps\")\n    out = torch.empty_like(x)\n    add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)\n    torch.mps.synchronize()\n    return (out - (x + y)).abs().max().item()\n"
  },
  {
    "id": "fused_relu",
    "label": "Fused ReLU x scale",
    "source": "import torch, triton, triton.language as tl\n\n@triton.jit\ndef relu_mul_kernel(x_ptr, s_ptr, out_ptr, n, BLOCK: tl.constexpr):\n    pid = tl.program_id(0)\n    off = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = off < n\n    x = tl.load(x_ptr + off, mask=mask)\n    s = tl.load(s_ptr + off, mask=mask)\n    y = tl.where(x > 0, x, 0.0) * s\n    tl.store(out_ptr + off, y, mask=mask)\n\ndef run():\n    n = 4096\n    x = torch.randn(n, device=\"mps\")\n    s = torch.randn(n, device=\"mps\")\n    out = torch.empty_like(x)\n    relu_mul_kernel[(triton.cdiv(n, 256),)](x, s, out, n, BLOCK=256)\n    torch.mps.synchronize()\n    ref = torch.relu(x) * s\n    return (out - ref).abs().max().item()\n"
  },
  {
    "id": "softmax",
    "label": "Softmax (row)",
    "source": "import torch, triton, triton.language as tl\n\n@triton.jit\ndef softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):\n    row = tl.program_id(0)\n    cols = tl.arange(0, BLOCK)\n    mask = cols < n_cols\n    ptr = x_ptr + row * n_cols + cols\n    x = tl.load(ptr, mask=mask, other=-float(\"inf\"))\n    x = x - tl.max(x, axis=0)\n    num = tl.exp(x)\n    den = tl.sum(num, axis=0)\n    tl.store(out_ptr + row * n_cols + cols, num / den, mask=mask)\n\ndef run():\n    rows, cols = 128, 512\n    x = torch.randn(rows, cols, device=\"mps\")\n    out = torch.empty_like(x)\n    BLOCK = triton.next_power_of_2(cols)\n    softmax_kernel[(rows,)](x, out, cols, BLOCK=BLOCK)\n    torch.mps.synchronize()\n    ref = torch.softmax(x, dim=1)\n    return (out - ref).abs().max().item()\n"
  },
  {
    "id": "conv1d",
    "label": "1D convolution",
    "source": "import torch, triton, triton.language as tl\n\n@triton.jit\ndef conv1d_kernel(x_ptr, w_ptr, out_ptr, n, K: tl.constexpr, BLOCK: tl.constexpr):\n    pid = tl.program_id(0)\n    off = pid * BLOCK + tl.arange(0, BLOCK)\n    acc = tl.zeros((BLOCK,), dtype=tl.float32)\n    for k in tl.static_range(K):\n        idx = off + k\n        x = tl.load(x_ptr + idx, mask=idx < n + K - 1, other=0.0)\n        acc += x * tl.load(w_ptr + k)\n    tl.store(out_ptr + off, acc, mask=off < n)\n\ndef run():\n    n, K = 4096, 5\n    x = torch.randn(n + K - 1, device=\"mps\")\n    w = torch.randn(K, device=\"mps\")\n    out = torch.empty(n, device=\"mps\")\n    conv1d_kernel[(triton.cdiv(n, 256),)](x, w, out, n, K=K, BLOCK=256)\n    torch.mps.synchronize()\n    ref = torch.stack([(x[i:i + K] * w).sum() for i in range(n)])\n    return (out - ref).abs().max().item()\n"
  },
  {
    "id": "layernorm",
    "label": "LayerNorm (row)",
    "source": "import torch, triton, triton.language as tl\n\n@triton.jit\ndef layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):\n    row = tl.program_id(0)\n    cols = tl.arange(0, BLOCK)\n    mask = cols < n_cols\n    ptr = x_ptr + row * n_cols + cols\n    x = tl.load(ptr, mask=mask, other=0.0)\n    mean = tl.sum(x, axis=0) / n_cols\n    xc = tl.where(mask, x - mean, 0.0)\n    var = tl.sum(xc * xc, axis=0) / n_cols\n    xhat = xc / tl.sqrt(var + eps)\n    w = tl.load(w_ptr + cols, mask=mask)\n    b = tl.load(b_ptr + cols, mask=mask)\n    tl.store(out_ptr + row * n_cols + cols, xhat * w + b, mask=mask)\n\ndef run():\n    rows, cols = 256, 768\n    x = torch.randn(rows, cols, device=\"mps\")\n    w = torch.randn(cols, device=\"mps\")\n    b = torch.randn(cols, device=\"mps\")\n    out = torch.empty_like(x)\n    BLOCK = triton.next_power_of_2(cols)\n    layernorm_kernel[(rows,)](x, w, b, out, cols, 1e-5, BLOCK=BLOCK)\n    torch.mps.synchronize()\n    ref = torch.nn.functional.layer_norm(x, (cols,), w, b, 1e-5)\n    return (out - ref).abs().max().item()\n"
  }
];
