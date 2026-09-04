//! 八叉树切块（AABB + 递归细分）
//!
//! 用于 3D Tiles hierarchical LOD 的切块策略。
//! 同一空间区域在不同深度对应不同 LOD level，深度的几何精度自下而上递增。

/// 轴对齐包围盒
#[derive(Debug, Clone)]
pub struct Aabb {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
        }
    }

    /// 包含一个点（闭区间）
    pub fn expand_point(&mut self, p: [f64; 3]) {
        for i in 0..3 {
            if p[i] < self.min[i] {
                self.min[i] = p[i];
            }
            if p[i] > self.max[i] {
                self.max[i] = p[i];
            }
        }
    }

    pub fn center(&self) -> [f64; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// 切分为 8 个子 AABB（按中心点平面分割）
    ///
    /// 子节点索引位编码：(x_bit, y_bit, z_bit) —— 位 0 = x 左/右，位 1 = y 下/上，位 2 = z 前/后
    pub fn split(&self) -> [Aabb; 8] {
        let c = self.center();
        let mut out = [(); 8].map(|_| self.clone());

        for i in 0..8 {
            out[i] = Aabb {
                min: [
                    if i & 1 == 0 { self.min[0] } else { c[0] },
                    if i & 2 == 0 { self.min[1] } else { c[1] },
                    if i & 4 == 0 { self.min[2] } else { c[2] },
                ],
                max: [
                    if i & 1 == 0 { c[0] } else { self.max[0] },
                    if i & 2 == 0 { c[1] } else { self.max[1] },
                    if i & 4 == 0 { c[2] } else { self.max[2] },
                ],
            };
        }
        out
    }

    pub fn contains_point(&self, p: &[f64; 3]) -> bool {
        p[0] >= self.min[0] && p[0] <= self.max[0] &&
        p[1] >= self.min[1] && p[1] <= self.max[1] &&
        p[2] >= self.min[2] && p[2] <= self.max[2]
    }

    /// 水平方向 (X, Y) 长度最大值
    /// 用于 octree 终止条件 —— 避免垂直方向被切得过细（薄片豆腐）
    pub fn horizontal_extent(&self) -> f64 {
        let ex = self.max[0] - self.min[0];
        let ey = self.max[1] - self.min[1];
        ex.max(ey)
    }

    /// 各轴长度最大值（用于盒子几何 / 报告打印）
    pub fn max_extent(&self) -> f64 {
        let mut m = 0.0_f64;
        for i in 0..3 {
            let e = self.max[i] - self.min[i];
            if e > m {
                m = e;
            }
        }
        m
    }
}

/// 八叉树节点
#[derive(Debug)]
pub struct OctreeNode {
    pub bounds: Aabb,
    /// 该节点持有的 mesh ID（仅叶节点非空）
    pub mesh_ids: Vec<usize>,
    /// 子节点（仅中间节点非空，叶节点为空 Vec）
    pub children: Vec<Box<OctreeNode>>,
    pub depth: u32,
}

impl OctreeNode {
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// 整棵树的叶节点数
    pub fn leaf_count(&self) -> usize {
        if self.is_leaf() {
            1
        } else {
            self.children.iter().map(|c| c.leaf_count()).sum()
        }
    }

    /// 整棵树覆盖的 mesh 总数（应等于输入 mesh 数）
    pub fn total_mesh_count(&self) -> usize {
        if self.is_leaf() {
            self.mesh_ids.len()
        } else {
            self.children.iter().map(|c| c.total_mesh_count()).sum()
        }
    }

    /// 按深度统计节点数
    pub fn depth_histogram(&self, out: &mut std::collections::HashMap<u32, usize>) {
        *out.entry(self.depth).or_insert(0) += 1;
        for c in &self.children {
            c.depth_histogram(out);
        }
    }

    /// 收集本节点**整棵子树**的所有 mesh ID（含后代）
    ///
    /// 中间节点的 `mesh_ids` 是空的，做多级 LOD 时需要知道自己子树下挂了哪些 mesh，
    /// 才能生成该层级的粗模型。
    pub fn collect_subtree_mesh_ids(&self, out: &mut Vec<usize>) {
        if self.is_leaf() {
            out.extend_from_slice(&self.mesh_ids);
        } else {
            for c in &self.children {
                c.collect_subtree_mesh_ids(out);
            }
        }
    }

    /// 收集所有叶节点
    pub fn collect_leaves<'a>(&'a self, out: &mut Vec<&'a OctreeNode>) {
        if self.is_leaf() {
            out.push(self);
        } else {
            for c in &self.children {
                c.collect_leaves(out);
            }
        }
    }
}

/// 八叉树构建器
pub struct OctreeBuilder {
    pub max_depth: u32,
    pub max_meshes_per_leaf: usize,
    pub min_extent: f64,
    /// 每 tile 的三角形预算上限。预算 = 目标 tile 大小上限 ×2（CLI --tile-size-mb
    /// 默认 10 → 容忍 ~20MB/tile 才切）。估算 b3dm 体积 = 三角形数 × 63B（实测校准）。
    pub max_triangles_per_tile: u64,
}

impl OctreeBuilder {
    /// 构建八叉树
    ///
    /// `mesh_centers[i]` 是第 i 个 mesh 的中心点（用于归属判断）
    /// `mesh_tri_counts[i]` 是第 i 个 mesh 的三角形数（fan 拆分后，用于体积预算）
    pub fn build(
        &self,
        root_bounds: Aabb,
        mesh_centers: &[[f64; 3]],
        mesh_tri_counts: &[u64],
    ) -> OctreeNode {
        let all_ids: Vec<usize> = (0..mesh_centers.len()).collect();
        self.build_recursive(root_bounds, all_ids, mesh_centers, mesh_tri_counts, 0)
    }

    fn build_recursive(
        &self,
        bounds: Aabb,
        mesh_ids: Vec<usize>,
        mesh_centers: &[[f64; 3]],
        mesh_tri_counts: &[u64],
        depth: u32,
    ) -> OctreeNode {
        // 体积估算：本节点挂的 mesh 的三角形总数（fan 后），按 63B/三角形 折算 b3dm 字节
        let est_tri: u64 = mesh_ids.iter().map(|&id| mesh_tri_counts[id]).sum();

        // 终止条件（任一满足即停）：
        //   1. 达最大深度（硬上限，防过度细分）
        //   2. 只剩 1 个 mesh（单 mesh 不可拆，再切也只是空转）
        //   3. mesh 数 ≤ 阈值 且 估算体积 ≤ 预算（目标 tile 大小，默认对应 ~10MB）
        //   4. 估算体积 ≤ 预算（体积达标直接停，不再按 mesh 数空切）
        //   5. 水平范围已经足够小（防"薄片豆腐"，2m 以下不再切）
        //
        // 【体积预算替代 min_extent=50 的原因】小范围高密度模型（如 ABC831 水平
        // 范围 <50m、245 万三角形）会被旧的 min_extent 一刀切成"整个模型一个
        // 156MB tile"。预算策略按字节控制，任何尺度的模型都能落到 2-20MB/tile。
        let stop = depth >= self.max_depth
            || mesh_ids.len() <= 1
            || (mesh_ids.len() <= self.max_meshes_per_leaf
                && est_tri <= self.max_triangles_per_tile)
            || est_tri <= self.max_triangles_per_tile
            || bounds.horizontal_extent() < self.min_extent;

        if stop {
            return OctreeNode {
                bounds,
                mesh_ids,
                children: Vec::new(),
                depth,
            };
        }

        // 切分
        let child_bounds = bounds.split();
        let mut child_meshes: [Vec<usize>; 8] = Default::default();

        // 分配：先按中心点归属子节点
        let mut unassigned: Vec<usize> = Vec::new();
        for &id in &mesh_ids {
            let center = mesh_centers[id];
            if let Some(idx) = (0..8).find(|&i| child_bounds[i].contains_point(&center)) {
                child_meshes[idx].push(id);
            } else {
                unassigned.push(id);
            }
        }

        // 处理中心点在切分面上的 mesh（数值边界情况）—— 归到距离最近的子节点
        for id in unassigned {
            let center = mesh_centers[id];
            let mut best_idx = 0;
            let mut best_dist = f64::INFINITY;
            for i in 0..8 {
                let cc = child_bounds[i].center();
                let dx = center[0] - cc[0];
                let dy = center[1] - cc[1];
                let dz = center[2] - cc[2];
                let d = dx * dx + dy * dy + dz * dz;
                if d < best_dist {
                    best_dist = d;
                    best_idx = i;
                }
            }
            child_meshes[best_idx].push(id);
        }

        // 递归构建子节点
        let children: Vec<Box<OctreeNode>> = (0..8)
            .map(|i| {
                Box::new(self.build_recursive(
                    child_bounds[i].clone(),
                    std::mem::take(&mut child_meshes[i]),
                    mesh_centers,
                    mesh_tri_counts,
                    depth + 1,
                ))
            })
            .collect();

        // 退化检查：如果某个子节点是空叶（mesh 数 0），它没意义但仍占一个 tile 槽
        // 暂时保留，由 B1 阶段过滤掉空 tile

        OctreeNode {
            bounds,
            mesh_ids: Vec::new(),
            children,
            depth,
        }
    }
}