# _3dtile

自研 FBX → 3D Tiles (b3dm) 切片工具。Rust 1.97 + ufbx 0.11（pure Rust crate，无 vcpkg/CMake 依赖）。

替代 CesiumLab model2tiles 的可控切片流水线，专为高密度工业模型（湛江油库 ABC831：180MB / 2217 mesh / 245万 三角形）设计。

## 快速上手

```bash
# 编译
cargo build --release

# 切片 ABC831.fbx 到 out 目录（湛江油库锚点 + -178° 航向，与 demo 配套）
D:/fbx_convert/_3dtile/target/release/_3dtile.exe build <FBX> <输出目录> \
  --longitude 110.43530332 --latitude 21.1977361 --heading=-178

# 切片后体检（GLB 结构、_BATCHID 值域、material 纹理索引、Batch Table 行数）
python tools/validate_tiles.py <输出目录>
```

## 核心能力

- **FBX→b3dm 全链路**：单位自适应（米/厘米/英寸）、坐标轴、per-face 材质、UV 真值、NORMAL 真值
- **pick 属性**（对齐 CesiumLab mainview）：per-vertex `_BATCHID` (FLOAT 5126) + Batch Table + scenetree.json
- **体积预算切块**：八叉树 + 三角形数预算（默认 20MB/tile）+ <2MB leaf 自动合并到空间最近大 leaf
- **贴图压缩**：最长边 256（GPU 占用 240 张 unique 贴图 ~60MB），含真 alpha 才留 PNG
- **GPU 优化**：sampler mipmap、AABB 真实顶点（防八叉树格子切外伸几何）

## 排错纪律（踩过的坑，已沉淀）

1. **单变量实验必须真正单变量**——不要同时改多参数排查
2. **优先 diff 参考实现**（如 CesiumLab mainview）而不是反复猜参数
3. **多轮 Edit 开关功能后必查残留**：`grep -n "indices"` 确认 primitive JSON 里没有重复键
4. **指标 ≠ 真实显示**：tile 加载数 / 0 重 fetch / nonBgRatio 高都不等于模型真的被画出来，**截图实锤**

## 坑清单（详见 skill）

- `_BATCHID` 必须是 **FLOAT (5126)**，UINT (5125) 让 Cesium 1.95 解析失败丢弃整个 primitive
- 不用 RTC_CENTER 键（mainview 不带）；带 `_BATCHID` 时与多 primitive 路径冲突
- indices 字段**不得与 _BATCHID 共用 bufferView**（重复键 bug 残留会让 indices 指向 batchId 数据）
- material.baseColorTexture.index 必须用 **textures 数组索引**（不是 images）
- glTF nodes 数 = meshes 数（mesh 必须经 node 桥接才能渲染）
- glTF texture 必须写 sampler（mipmap）

## 目录

- `src/` — Rust 源码（main.rs + b3dm.rs）
- `tools/validate_tiles.py` — 切片产物体检脚本（b3dm/GLB 结构 + 引用一致性）
- `.gitignore` — Rust 工具惯例：target/ + Cargo.lock 不入库