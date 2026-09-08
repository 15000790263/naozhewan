//! b3dm 1.0 + glTF 2.0 (GLB) 写入器
//!
//! b3dm 格式：[b3dm header 28B] + [Feature Table JSON] + [Batch Table JSON] + [GLB payload]
//! GLB 格式：[glTF header 12B] + [JSON chunk] + [BIN chunk]
//!
//! 当前版本：P0 — 不做 Draco 压缩，端到端格式正确即可

use anyhow::{Context, Result};
use serde_json::{json, Value};

/// 单个 mesh 的几何数据（从 ufbx 提取）
#[derive(Debug, Clone)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    /// 三角形索引（每 3 个一组）
    pub indices: Vec<u32>,
    pub name: String,
    /// 节点变换含反射时，是否需要翻转三角形绕序。
    pub flip_winding: bool,
    /// 贴图唯一键（仅用于同一 tile 内合并材质）
    pub texture_uri: Option<String>,
    /// 贴图原始二进制，嵌入 GLB BIN chunk；None 表示无贴图。
    pub texture_bytes: Option<Vec<u8>>,
    /// 贴图 MIME 类型，例如 image/png / image/jpeg。
    pub texture_mime: Option<String>,
    /// 该贴图是否含有效 alpha 通道。
    pub texture_has_alpha: bool,
    /// PBR 金属度（FBX material.pbr.metalness.value_vec4.x，未设置时默认 0.0）
    pub metallic_factor: f32,
    /// PBR 粗糙度（FBX material.pbr.roughness.value_vec4.x，未设置时默认 1.0）
    pub roughness_factor: f32,
    /// 所属构件的全局 mesh 索引（FBX scene.meshes 下标）。
    /// pick 属性粒度 = 构件（mesh）：同一 mesh 拆出的多个材质组 part 共享同一 feature_gid。
    pub feature_gid: usize,
    /// 每顶点的 batchId（tile 内构件序号），与 positions 等长。
    /// 用于 b3dm 的 _BATCHID attribute：Cesium 靠它把点击像素映射到 Batch Table 行。
    /// pick 功能开启时必填；关闭时留空 Vec。
    pub batch_ids: Vec<u32>,
}

/// 一组 mesh（一个 leaf node 下的所有 mesh 合并）
pub struct TileGeometry {
    pub meshes: Vec<MeshData>,
    /// tile 内构件名表（下标 = batchId）。Batch Table 的 name 行按此生成，
    /// 行数 = 构件数（而不是合并后的 primitive 数——那会导致 pick 属性错位）。
    pub batch_names: Vec<String>,
    /// 构件的全局 mesh 索引（下标 = batchId）。用于生成跨 tile 一致的构件 id
    /// （Batch Table 与 scenetree.json 的 id 同算法同输入，页面可互查高亮）。
    pub batch_gids: Vec<usize>,
}

impl TileGeometry {
    /// 总顶点数 / 三角形数
    pub fn total_vertices(&self) -> usize {
        self.meshes.iter().map(|m| m.positions.len()).sum()
    }
    pub fn total_triangles(&self) -> usize {
        self.meshes.iter().map(|m| m.indices.len() / 3).sum()
    }

    /// 合并所有 mesh 成一个 primitive（所有顶点拼到一起，索引加偏移）
    pub fn merged(&self) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u32>) {
        let total_v = self.total_vertices();
        let total_i: usize = self.meshes.iter().map(|m| m.indices.len()).sum();
        let mut positions = Vec::with_capacity(total_v);
        let mut normals = Vec::with_capacity(total_v);
        let mut uvs = Vec::with_capacity(total_v);
        let mut indices = Vec::with_capacity(total_i);
        let mut offset = 0u32;
        for m in &self.meshes {
            positions.extend_from_slice(&m.positions);
            normals.extend_from_slice(&m.normals);
            uvs.extend_from_slice(&m.uvs);
            for &idx in &m.indices {
                indices.push(idx + offset);
            }
            offset += m.positions.len() as u32;
        }
        (positions, normals, uvs, indices)
    }

    /// 计算本 tile 全部 mesh 顶点的真实 AABB（min, max）
    ///
    /// 【boundingVolume 不准的修复】八叉树是按 mesh 中心点切分的，大 mesh 的顶点
    /// 会伸出其所属格子之外。tileset.json 若直接用八叉树格子的 AABB 做 boundingVolume，
    /// 包围盒会小于实际几何 → Cesium 视锥剔除误判 → 相机贴近时整块 tile 消失。
    /// 所以必须用**真实顶点**算 AABB。
    pub fn compute_aabb_all(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        let mut any = false;
        for m in &self.meshes {
            for p in &m.positions {
                any = true;
                for i in 0..3 {
                    if p[i] < min[i] { min[i] = p[i]; }
                    if p[i] > max[i] { max[i] = p[i]; }
                }
            }
        }
        if any { Some((min, max)) } else { None }
    }

    /// 计算 AABB
    pub fn compute_aabb(positions: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for p in positions {
            for i in 0..3 {
                if p[i] < min[i] {
                    min[i] = p[i];
                }
                if p[i] > max[i] {
                    max[i] = p[i];
                }
            }
        }
        (min, max)
    }
}

/// 顶点聚类简化（vertex clustering）—— 多级 LOD 的粗层生成算法
///
/// 原理：把空间切成 `cell_size` 的立方体网格，落进同一格的顶点合并成一个，
/// 三角形三个顶点若落进同一格（或两格）就退化被丢弃。
/// 这是最简单、最快、且**保持拓扑边界**的简化算法，非常适合远景 LOD。
///
/// `cell_size <= 0` 时原样返回（LOD0 不简化）。
///
/// ⚠️ 只合并位置落在同一格的顶点，**不跨格做平均**——否则会让模型表面收缩。
/// 简化 mesh。
///
/// 参数 `keep_ratio`：保留的三角形比例（0.0~1.0）。>= 1.0 表示不简化。
///
/// 【为什么用比例而不是误差阈值驱动】
///   LOD 金字塔里粗层节点天然包含"整棵子树的全部几何"（根节点 = 全模型），
///   用"相对误差"驱动时，误差设小了减不动（实测 L0 仍有 120MB），设大了形状崩。
///   而 Cesium ion / CesiumLab 的粗层是**目标三角形数明确**的（几万级别，
///   保证远景秒开）。所以改成按层给比例，配合一个宽松的误差上限兜底防崩。
pub fn simplify_mesh(m: &MeshData, keep_ratio: f32) -> MeshData {
    if keep_ratio >= 1.0 || keep_ratio <= 0.0 || m.positions.is_empty() || m.indices.is_empty() {
        return m.clone();
    }

    // ===== meshoptimizer QEM 简化 =====
    //
    // 为什么放弃顶点分桶（cell clustering）：
    //   分桶按"顶点落在哪个网格格子"合并，格子一大就把地面/墙面整片糊掉
    //   （用户描述"像狗咬了一部分"），而且合并后顶点序变了，batch_ids 需要
    //   remap，容易出错。
    //
    // QEM（Quadric Edge Collapse）折叠边时用二次误差矩阵评估"引入多少形状误差"，
    // 优先折叠平面区域、保留边界与尖锐特征 —— 远景是"变简单"而不是"变破"。
    // 更重要的是 simplify() 返回的索引**引用原顶点**，positions/normals/uvs/
    // batch_ids 全部不动，pick 属性天然正确（不需要 remap）。

    // 顶点数据字节视图：stride=12（3×f32），position_offset=0
    let byte_slice: &[u8] = unsafe {
        std::slice::from_raw_parts(
            m.positions.as_ptr().cast::<u8>(),
            m.positions.len() * std::mem::size_of::<[f32; 3]>(),
        )
    };
    let adapter = match meshopt::VertexDataAdapter::new(byte_slice, 12, 0) {
        Ok(a) => a,
        Err(_) => return m.clone(),
    };

    // target_count = **目标索引数**（不是三角形数）：按 keep_ratio 折算。
    // target_error 给一个宽松的相对误差上限（mesh 尺寸的 5%）兜底：
    // 比例达标但形状误差超限时 meshopt 会停止，避免模型崩坏。
    let target_count = ((m.indices.len() as f32 * keep_ratio) as usize / 3 * 3).max(12);
    // 【关键】target_error 必须按层放宽：粗层要减到 2%，形状误差必然很大。
    //   早先固定 0.05 时，meshopt 还没减到目标数就被误差上限叫停
    //   （实测比例改到 0.02 后体积仍是 577MB，等于没简化）。
    //   粗层（比例小）允许大误差，细层（比例大）保持严格。
    let target_error = (0.05 / keep_ratio).min(1.0);
    let mut result_error = 0.0f32;

    let mut new_indices = meshopt::simplify::simplify(
        &m.indices,
        &adapter,
        target_count,
        target_error,
        // 【关键】Permissive：允许跨属性不连续（UV 接缝/硬边）折叠。
        //   我们的顶点是 (pos,nrm,uv) 三元组，几乎每个顶点都在缝上——
        //   默认模式下它们全部不可折叠，实测无论 target_count 给多少都只减 ~50%。
        //   粗层是远景，缝混合的贴图偏差肉眼不可见，Permissive 是正确取舍。
        meshopt::simplify::SimplifyOptions::Permissive,
        Some(&mut result_error),
    );

    // simplify 失败/退化时退回原 mesh（打日志统计发生率）
    if new_indices.is_empty() || new_indices.len() >= m.indices.len() {
        eprintln!(
            "[lod-skip] {} 索引 {} 未简化 (meshopt 返回 {})",
            m.name, m.indices.len(), new_indices.len()
        );
        return m.clone();
    }

    // ===== 顶点压实 =====
    //
    // simplify 只换索引，被折叠掉的顶点仍留在数组里 → 体积几乎不降（实测
    // 58 tile 从 192MB 涨到 693MB）。必须压实掉"不再被任何索引引用"的顶点。
    //
    // 做法：把 pos/nrm/uv/batch_id 打包成一个 #[repr(C)] 结构体，交给
    // optimize_vertex_fetch 一起压实 —— 四个数组同步重排，batch_ids 自动正确
    // （不需要手写 remap，也就不存在 remap 写错导致 pick 属性错位的风险）。
    let vtxs: Vec<SimplifyVertex> = (0..m.positions.len())
        .map(|i| SimplifyVertex {
            pos: m.positions[i],
            nrm: if i < m.normals.len() { m.normals[i] } else { [0.0, 1.0, 0.0] },
            uv: if i < m.uvs.len() { m.uvs[i] } else { [0.0, 0.0] },
            bid: m.batch_ids.get(i).copied().unwrap_or(0),
        })
        .collect();

    let packed = meshopt::optimize::optimize_vertex_fetch(&mut new_indices, &vtxs);

    let mut positions = Vec::with_capacity(packed.len());
    let mut normals = Vec::with_capacity(packed.len());
    let mut uvs = Vec::with_capacity(packed.len());
    let mut batch_ids = Vec::with_capacity(packed.len());
    for v in packed {
        positions.push(v.pos);
        normals.push(v.nrm);
        uvs.push(v.uv);
        batch_ids.push(v.bid);
    }

    MeshData {
        positions,
        normals,
        uvs,
        indices: new_indices,
        name: m.name.clone(),
        texture_uri: m.texture_uri.clone(),
        texture_bytes: m.texture_bytes.clone(),
        texture_mime: m.texture_mime.clone(),
        texture_has_alpha: m.texture_has_alpha,
        feature_gid: m.feature_gid,
        batch_ids,
        flip_winding: m.flip_winding,
        metallic_factor: m.metallic_factor,
        roughness_factor: m.roughness_factor,
    }
}

/// 简化/压实用的打包顶点：pos + nrm + uv + batchId 一起重排，
/// 保证几何属性与 pick 属性（batchId）永远对齐。
#[repr(C)]
#[derive(Clone, Default)]
struct SimplifyVertex {
    pos: [f32; 3],
    nrm: [f32; 3],
    uv: [f32; 2],
    bid: u32,
}

/// 顶点云的外接球半径（用于把米为单位的误差转成 meshopt 需要的相对误差）
fn mesh_radius(positions: &[[f32; 3]]) -> f32 {
    let mut mnx = [f32::INFINITY; 3];
    let mut mxx = [f32::NEG_INFINITY; 3];
    for p in positions {
        for k in 0..3 {
            if p[k] < mnx[k] {
                mnx[k] = p[k];
            }
            if p[k] > mxx[k] {
                mxx[k] = p[k];
            }
        }
    }
    let dx = (mxx[0] - mnx[0]) as f64;
    let dy = (mxx[1] - mnx[1]) as f64;
    let dz = (mxx[2] - mnx[2]) as f64;
    (0.5 * (dx * dx + dy * dy + dz * dz).sqrt()) as f32
}

/// 把同一 tile 内共享同一贴图（texture_uri 相同）的 mesh 合并成一个大 primitive。
///
/// 【为什么必须合并】Cesium 对 glTF 的每个 primitive 各发一次 draw call。per-face
/// material 拆分会把 primitive 数放大到与材质片断同量级（本模型 4111 个，而
/// CesiumLab 同模型只有 192 个），draw call 数量是帧率的第一杀手——这才是
/// 页面卡顿/低帧率的真根因（贴图与体积反而不是：CesiumLab 304MB 都比我们流畅）。
///
/// 合并安全性：同 tile 内同贴图的 mesh 顶点都已烘焙到同一世界坐标、属性布局一致
/// （pos+nrm+uv 严格等长），可以无损拼接：索引平移 +vbase；flip_winding 的 mesh
/// 提前把三角绕序翻好，合并结果统一为非 flip（flip_winding=false）。
/// 渲染结果与拆分时逐字节一致，只是 draw call 从 N 降到 1。
pub fn merge_meshes_by_texture(meshes: Vec<MeshData>) -> Vec<MeshData> {
    // 组顺序 = 首见顺序；键 = texture_uri（None 视为"无贴图"一组）
    let mut groups: Vec<Option<String>> = Vec::new();
    let mut merged: Vec<MeshData> = Vec::new();
    for m in meshes {
        let key = m.texture_uri.clone();
        match groups.iter().position(|g| *g == key) {
            Some(gi) => {
                let dst = &mut merged[gi];
                let vbase = dst.positions.len() as u32;
                dst.positions.extend_from_slice(&m.positions);
                dst.normals.extend_from_slice(&m.normals);
                dst.uvs.extend_from_slice(&m.uvs);
                dst.batch_ids.extend_from_slice(&m.batch_ids);
                if m.flip_winding {
                    for t in m.indices.chunks_exact(3) {
                        dst.indices.push(t[0] + vbase);
                        dst.indices.push(t[2] + vbase);
                        dst.indices.push(t[1] + vbase);
                    }
                } else {
                    dst.indices.extend(m.indices.iter().map(|&i| i + vbase));
                }
                dst.name.push('+');
                dst.name.push_str(&m.name);
            }
            None => {
                groups.push(key);
                merged.push(m);
            }
        }
    }
    for mm in &mut merged {
        mm.flip_winding = false;
    }
    merged
}

/// 检测贴图文件是否含 alpha 通道（用于决定 glTF 材质的 alphaMode）
///
/// 只解析 PNG 文件头（IHDR color type + tRNS chunk），不解码像素：
/// - color type 4（gray+A）/ 6（RGBA）→ 有 alpha 通道
/// - color type 3（palette）且存在 tRNS chunk → 有透明色
/// - JPG / 其它格式 → 无 alpha
///
/// 注意：RGBA 但 alpha 全 255 的图会被"误判"为有 alpha，但 MASK(cutoff=0.5)
/// 在 alpha=1 时等价 OPAQUE，无视觉副作用，所以不需要更精确的像素级检测。
pub fn texture_file_has_alpha(path: &std::path::Path) -> bool {
    let mut buf = [0u8; 64];
    let n = match std::fs::File::open(path) {
        Ok(mut f) => match std::io::Read::read(&mut f, &mut buf) {
            Ok(n) => n,
            Err(_) => return false,
        },
        Err(_) => return false,
    };
    if n < 26 || &buf[0..8] != b"\x89PNG\r\n\x1a\n" {
        return false;
    }
    // IHDR 数据从偏移 16 开始，color type 在第 9 字节（偏移 24）
    let color_type = buf[25];
    match color_type {
        4 | 6 => true,
        3 => {
            // palette：要扫 chunk 列表找 tRNS（最多读 4KB，tRNS 通常紧跟 IHDR）
            let mut head = vec![0u8; 4096];
            let m = std::fs::File::open(path)
                .and_then(|mut f| std::io::Read::read(&mut f, &mut head))
                .unwrap_or(0);
            head.truncate(m);
            // 从偏移 8 开始遍历 chunk：len(4) type(4) data(len) crc(4)
            let mut off = 8usize;
            while off + 8 <= head.len() {
                let len =
                    u32::from_be_bytes([head[off], head[off + 1], head[off + 2], head[off + 3]])
                        as usize;
                let ctype = &head[off + 4..off + 8];
                if ctype == b"tRNS" {
                    return true;
                }
                if ctype == b"IDAT" {
                    // tRNS 必须在第一个 IDAT 之前
                    return false;
                }
                off += 12 + len;
            }
            false
        }
        _ => false,
    }
}

/// 把 GLB 写入到 bytes（glTF 2.0 JSON + BIN 二进制）
///
/// 每个 mesh 保留独立几何+贴图，多个 mesh 共用同一贴图时合并为同一个 material，
/// 保证每个 mesh 都有正确的贴图（草地绿/罐子白/地面灰各取各的）。
///
/// glTF 结构：
/// - images[i] / textures[i] / materials[i]：每个唯一贴图对应一条
/// - meshes[i].primitives[0]：每个 mesh 一个 primitive，引用自己的 material index
pub fn build_glb(geometry: &TileGeometry) -> Result<Vec<u8>> {
    if geometry.total_vertices() == 0 || geometry.total_triangles() == 0 {
        anyhow::bail!("TileGeometry 为空");
    }

    // ---- 1. 按 unique 贴图分组 mesh，每个 unique 贴图分配 material index ----
    // texture_uri -> material index（共用同一贴图 + 同一 PBR 参数的 mesh 共享 material）
    // 注意：合并键包含 metallic + roughness，否则同一贴图的不同 mesh 实例会被
    // 强制用同一粗糙度，导致有的实例反射对、有的反射错（FBX 里同一贴图常用于
    // 多个 roughness 不同的子部件：油罐本体 metallic=0 roughness=0.45，扶手
    // metallic=0 roughness=0.7，反射观感不一样）。
    let mut unique_textures: Vec<Option<String>> = Vec::new();
    // material key = (texture_uri, metallic, roughness)
    let mut mat_keys: Vec<(Option<String>, f32, f32)> = Vec::new();
    let mut tex_to_mat: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (mesh_idx, mesh) in geometry.meshes.iter().enumerate() {
        let key = (mesh.texture_uri.clone(), mesh.metallic_factor, mesh.roughness_factor);
        let mat_idx = match mat_keys.iter().position(|k| k == &key) {
            Some(p) => p,
            None => {
                // 注册 image（如果还没注册）
                let img_idx = if let Some(uri) = &mesh.texture_uri {
                    match unique_textures.iter().position(|t| t.as_ref() == Some(uri)) {
                        Some(p) => p,
                        None => {
                            let p = unique_textures.len();
                            unique_textures.push(Some(uri.clone()));
                            p
                        }
                    }
                } else {
                    // 无贴图 mesh 共享一个 None image 占位
                    match unique_textures.iter().position(|t| t.is_none()) {
                        Some(p) => p,
                        None => {
                            let p = unique_textures.len();
                            unique_textures.push(None);
                            p
                        }
                    }
                };
                let _ = img_idx;
                let p = mat_keys.len();
                mat_keys.push(key);
                p
            }
        };
        tex_to_mat.insert(mesh_idx, mat_idx);
    }

    // ---- 2. 组装 BIN：每个 mesh 独立的顶点段 ----
    // layout: mesh0 pos+nrm+uv+idx, mesh1 pos+nrm+uv+idx, ...
    // accessor[i*4+0..i*4+3] 对应 mesh i 的 pos/nrm/uv/idx
    #[derive(Clone)]
    struct MeshBinSection {
        pos_off: usize, pos_len: usize,
        nrm_off: usize, nrm_len: usize,
        uv_off:  usize, uv_len:  usize,
        bid_off: usize, bid_len: usize,
        idx_off: usize, idx_len: usize,
        v_count: usize, i_count: usize,
        min: [f32; 3], max: [f32; 3],
    }
    let mut sections: Vec<MeshBinSection> = Vec::new();
    let mut bin_data: Vec<u8> = Vec::new();

    for mesh in &geometry.meshes {
        let pos_bytes: Vec<u8> = mesh.positions.iter()
            .flat_map(|p| p.iter().flat_map(|f| f.to_le_bytes())).collect();
        let nrm_bytes: Vec<u8> = mesh.normals.iter()
            .flat_map(|n| n.iter().flat_map(|f| f.to_le_bytes())).collect();
        let uv_bytes: Vec<u8> = mesh.uvs.iter()
            .flat_map(|u| u.iter().flat_map(|f| f.to_le_bytes())).collect();
        // _BATCHID：每顶点的构件序号。合并后的 primitive 内多个构件各带各的
        // batchId，Cesium 按像素命中的顶点 batchId 查 Batch Table 对应行。
        // 【关键】b3dm 1.0 的 _BATCHID 必须是 **FLOAT (5126)**——Cesium 1.95 对
        // UINT(5125) 类型的 _BATCHID 会解析失败导致整个 primitive 不渲染
        // （mainview/CesiumLab 就是 5126 float，单变量实验已验证）。
        let batch_bytes: Vec<u8> = if mesh.batch_ids.len() == mesh.positions.len() {
            mesh.batch_ids.iter().flat_map(|b| (*b as f32).to_le_bytes()).collect()
        } else {
            // 兜底：batch_ids 缺失时全部归 0（整个 primitive 一个 feature）
            vec![0u8; 4 * mesh.positions.len()]
        };
        let idx_bytes: Vec<u8> = if mesh.flip_winding {
            mesh.indices.chunks_exact(3)
                .flat_map(|t| [t[0], t[2], t[1]])
                .flat_map(|i| i.to_le_bytes())
                .collect()
        } else {
            mesh.indices.iter()
                .flat_map(|i| i.to_le_bytes())
                .collect()
        };

        let (min, max) = TileGeometry::compute_aabb(&mesh.positions);
        let pos_off = bin_data.len();
        bin_data.extend_from_slice(&pos_bytes);
        let nrm_off = bin_data.len();
        bin_data.extend_from_slice(&nrm_bytes);
        let uv_off = bin_data.len();
        bin_data.extend_from_slice(&uv_bytes);
        let bid_off = bin_data.len();
        bin_data.extend_from_slice(&batch_bytes);
        let idx_off = bin_data.len();
        bin_data.extend_from_slice(&idx_bytes);

        sections.push(MeshBinSection {
            pos_off, pos_len: pos_bytes.len(),
            nrm_off, nrm_len: nrm_bytes.len(),
            uv_off,  uv_len:  uv_bytes.len(),
            bid_off, bid_len: batch_bytes.len(),
            idx_off, idx_len: idx_bytes.len(),
            v_count: mesh.positions.len(),
            i_count: mesh.indices.len(),
            min, max,
        });
    }

    // 图片追加到同一个 BIN chunk，记录 image 对应的 bufferView。
    // 【坑】带 _BATCHID 时每 mesh 占 5 个 bufferView（pos/nrm/uv/batchid/idx），
    // image 起始索引必须 = sections.len() * 5，否则 image 指向几何区间报解码失败。
    let geometry_view_count = sections.len() * 5;
    let mut image_views: Vec<Value> = Vec::new();
    let mut image_info: Vec<(String, String, usize)> = Vec::new();
    for tex in &unique_textures {
        if let Some(name) = tex {
            if let Some(source) = geometry.meshes.iter().find(|m| m.texture_uri.as_deref() == Some(name.as_str())) {
                if let (Some(bytes), Some(mime)) = (&source.texture_bytes, &source.texture_mime) {
                    let pad = (4 - (bin_data.len() % 4)) % 4;
                    bin_data.extend(std::iter::repeat(0u8).take(pad));
                    let offset = bin_data.len();
                    bin_data.extend_from_slice(bytes);
                    image_info.push((name.clone(), mime.clone(), bytes.len()));
                    image_views.push(json!({
                        "buffer": 0,
                        "byteOffset": offset,
                        "byteLength": bytes.len()
                    }));
                }
            }
        }
    }

    // 图片 bufferView 必须追加到 glTF 的 bufferViews 数组。
    // image 的索引从 geometry_view_count 开始，若漏掉这一步，Cesium 会读取
    // bufferViews[24] 得到 undefined，最终报 Cannot read properties of undefined (reading 'buffer')。
    // 这里先保存 image_views，待 geometry bufferViews 组装后统一追加。

    // 4字节对齐 BIN
    let bin_pad = (4 - (bin_data.len() % 4)) % 4;
    bin_data.extend(std::iter::repeat(0u8).take(bin_pad));
    let total_bin = bin_data.len();

    // ---- 3. 组装 glTF JSON ----
    // 图片二进制稍后追加到同一个 BIN chunk，images 通过 bufferView 引用。
    let mut images: Vec<Value> = Vec::new();
    let mut textures: Vec<Value> = Vec::new();
    let mut materials: Vec<Value> = Vec::new();

    // sampler 0：所有贴图共用
    //   magFilter 9729 = LINEAR
    //   minFilter 9987 = LINEAR_MIPMAP_LINEAR（关键：启用 mipmap，消除缩小时的雪花噪点）
    //   wrapS/T  10497 = REPEAT（平铺，铺装类贴图必须）
    let samplers: Vec<Value> = vec![json!({
        "magFilter": 9729,
        "minFilter": 9987,
        "wrapS": 10497,
        "wrapT": 10497
    })];

    // uri -> 是否含 alpha（树/植物 billboard 贴图）。同一 uri 的 alpha 属性恒定，
    // 从各 mesh 的标志收集（extract_material_texture 拷贝贴图时已检测）。
    let mut uri_has_alpha: std::collections::HashMap<&str, bool> = std::collections::HashMap::new();
    for m in &geometry.meshes {
        if let Some(uri) = &m.texture_uri {
            uri_has_alpha.entry(uri.as_str()).or_insert(m.texture_has_alpha);
        }
    }

    // 先建 uri -> image/texture idx 的映射（unique_textures 索引 ≠ textures 索引：
    // images 含无贴图 mesh 的 {} 占位，textures 只有有贴图的才 push，两者会错位！
    // material.baseColorTexture.index 必须指向 textures 数组，所以用 uri_to_tex。）
    let mut uri_to_img: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut uri_to_tex: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for tex in &unique_textures {
        if let Some(uri) = tex {
            let img_idx = images.len();
            let info_idx = image_info.iter().position(|(name, _, _)| name == uri);
            if let Some(ii) = info_idx {
                images.push(json!({
                    "bufferView": geometry_view_count + ii,
                    "mimeType": image_info[ii].1
                }));
            } else {
                images.push(json!({ "uri": uri }));
            }
            // sampler 0 = mipmap 三线性过滤
            // 【贴图雪花点闪烁的修复】不写 sampler 时 glTF 默认 minFilter=LINEAR（无 mipmap），
            //   模型缩小/斜视时高频采样必然产生摩尔纹与雪花噪点。
            //   9987 = LINEAR_MIPMAP_LINEAR（mipmap 三线性），Cesium 会自动生成 mipmap 链。
            let tex_idx = textures.len();
            textures.push(json!({ "sampler": 0, "source": img_idx }));
            uri_to_img.insert(uri.clone(), img_idx);
            uri_to_tex.insert(uri.clone(), tex_idx);
        } else {
            // None 留空（占位，无 texture）
            images.push(json!({}));
        }
    }

    // 按 mat_keys 生成 material：每个 (texture_uri, metallic, roughness) 三元组一个 material
    // 【反射效果修复】之前硬编码 roughnessFactor=1.0（全漫反射）→ 罐体金属表面没有高光。
    //   FBX 自带 PBR 参数（实测 mainview 用的 roughness=0.45），现在从 ufbx::Material.pbr
    //   取出 metallic/roughness 写入 glTF，Cesium 1.95 PBR 管线就会按真实物理反射渲染。
    for (key_idx, (uri_opt, metallic, roughness)) in mat_keys.iter().enumerate() {
        let name = uri_opt.clone().unwrap_or_else(|| "default".to_string());
        let mut pbr = json!({
            "metallicFactor": metallic,
            "roughnessFactor": roughness,
        });
        if let Some(uri) = uri_opt {
            if let Some(&tex_idx) = uri_to_tex.get(uri) {
                pbr["baseColorTexture"] = json!({ "index": tex_idx });
            }
        } else {
            // 无贴图：白色 baseColor
            pbr["baseColorFactor"] = json!([1.0, 1.0, 1.0, 1.0]);
        }
        let mut mat = json!({
            "name": format!("{}_m{}", name, key_idx),
            "pbrMetallicRoughness": pbr
        });
        // 【透明通道修复】含 alpha 的贴图（树/栅栏植物 billboard）必须声明 alphaMode，
        //   否则 glTF 默认 OPAQUE，alpha=0 的区域被渲染成白色实心方块。
        //   用 MASK（硬阈值）而不是 BLEND：无排序问题、性能好，树叶边缘本来就是硬边。
        //   doubleSided=true：billboard 交叉面片从背面看也要可见，否则半棵树消失。
        if let Some(uri) = uri_opt {
            if uri_has_alpha.get(uri.as_str()).copied().unwrap_or(false) {
                mat["alphaMode"] = json!("MASK");
                mat["alphaCutoff"] = json!(0.5);
                mat["doubleSided"] = json!(true);
            }
        }
        materials.push(mat);
    }

    // meshes / accessors / bufferViews：每个 mesh 独立
    let mut meshes: Vec<Value> = Vec::new();
    let mut accessors: Vec<Value> = Vec::new();
    let mut buffer_views: Vec<Value> = Vec::new();

    for (i, sec) in sections.iter().enumerate() {
        let base = i * 5; // _BATCHID 恢复：5 个 bufferView（pos/nrm/uv/batchid/idx）

        buffer_views.push(json!({ "buffer": 0, "byteOffset": sec.pos_off, "byteLength": sec.pos_len, "target": 34962 }));
        buffer_views.push(json!({ "buffer": 0, "byteOffset": sec.nrm_off, "byteLength": sec.nrm_len, "target": 34962 }));
        buffer_views.push(json!({ "buffer": 0, "byteOffset": sec.uv_off,  "byteLength": sec.uv_len,  "target": 34962 }));
        buffer_views.push(json!({ "buffer": 0, "byteOffset": sec.bid_off, "byteLength": sec.bid_len, "target": 34962 }));
        buffer_views.push(json!({ "buffer": 0, "byteOffset": sec.idx_off, "byteLength": sec.idx_len, "target": 34963 }));

        accessors.push(json!({ "bufferView": base + 0, "componentType": 5126, "count": sec.v_count, "type": "VEC3", "min": sec.min, "max": sec.max }));
        accessors.push(json!({ "bufferView": base + 1, "componentType": 5126, "count": sec.v_count, "type": "VEC3" }));
        accessors.push(json!({ "bufferView": base + 2, "componentType": 5126, "count": sec.v_count, "type": "VEC2" }));
        accessors.push(json!({ "bufferView": base + 3, "componentType": 5126, "count": sec.v_count, "type": "SCALAR" }));
        accessors.push(json!({ "bufferView": base + 4, "componentType": 5125, "count": sec.i_count, "type": "SCALAR" }));

        let mat_idx = tex_to_mat[&i];
        meshes.push(json!({
            "name": geometry.meshes[i].name.clone(),
            "primitives": [{
                "attributes": {
                    "POSITION": base + 0,
                    "NORMAL":   base + 1,
                    "TEXCOORD_0": base + 2,
                    "_BATCHID": base + 3
                },
                "indices": base + 4,
                "material": mat_idx,
                "mode": 4  // TRIANGLES
            }]
        }));
    }

    // 图片数据已经追加到 BIN，必须把对应 bufferView 追加到 glTF bufferViews。
    buffer_views.extend(image_views);

    // nodes / scenes：每个 mesh 一个 node，全部挂到 scene[0] 下
    // 【P1 第二步修复】之前 P0 阶段误写成 nodes=[{mesh:0}]，导致每 tile 70 mesh
    //   只渲染了 mesh 0，69 个 mesh 全成了死数据（用户在 Cesium 里只看到"一点点"）。
    //   glTF 2.0 规范要求 mesh 必须经 node 才能被渲染树引用。
    let nodes: Vec<Value> = (0..geometry.meshes.len())
        .map(|i| json!({
            "mesh": i,
            "name": geometry.meshes[i].name.clone()
        }))
        .collect();
    let scene_nodes: Vec<usize> = (0..geometry.meshes.len()).collect();

    let gltf_json = json!({
        "asset": { "version": "2.0", "generator": "_3dtile 0.1.0" },
        "scene": 0,
        "scenes": [{ "nodes": scene_nodes }],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "textures": textures,
        "images": images,
        "samplers": samplers,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{ "byteLength": total_bin }]
    });

    let json_str = serde_json::to_string(&gltf_json).context("glTF JSON 序列化失败")?;
    let json_bytes = json_str.as_bytes();
    let json_pad = (4 - (json_bytes.len() % 4)) % 4;
    let json_chunk_len = json_bytes.len() + json_pad;

    // ---- 4. 拼装 GLB ----
    let glb_len = 12 + 8 + json_chunk_len + 8 + total_bin;
    let mut out = Vec::with_capacity(glb_len);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(glb_len as u32).to_le_bytes());
    out.extend_from_slice(&(json_chunk_len as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(json_bytes);
    out.extend(std::iter::repeat(b' ').take(json_pad));
    out.extend_from_slice(&(total_bin as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin_data);

    Ok(out)
}

/// 把 GLB 包成 b3dm 1.0
pub fn build_b3dm(geometry: &TileGeometry, _geometric_error: f64) -> Result<Vec<u8>> {
    let glb = build_glb(geometry)?;

    // Feature Table JSON（对齐 mainview/CesiumLab：只有 BATCH_LENGTH，不带 RTC_CENTER——
    // 顶点已经是相对 tileset 锚点的绝对局部坐标，RTC_CENTER 键会触发 Cesium 1.95
    // 不同的 batchId 处理路径，与 _BATCHID 组合时静默不渲染）
    let batch_len = geometry.batch_names.len().max(1);
    let ft = json!({
        "BATCH_LENGTH": batch_len
    });
    let ft_str = serde_json::to_string(&ft)?;
    let ft_bytes = ft_str.as_bytes();
    let ft_pad = (8 - (ft_bytes.len() % 8)) % 8;
    let ft_chunk_len = ft_bytes.len() + ft_pad;

    // Batch Table JSON（对齐 CesiumLab 风格）
    //
    // 行数 = 构件数（batchId 值域），不是合并后的 primitive 数：
    // 同一构件拆出的多个材质组 / 同贴图合并进同一 primitive 的多个构件，
    // 顶点各带各的 _BATCHID，pick 命中哪个构件就取哪一行。
    //
    // 字段：
    //   - _pack_props_: 每行的属性名（这里没有可打包数值，所以全是 "null"）
    //   - id:           每个构件的稳定 hash id（与 scenetree.json 的 id 同算法，可互查）
    //   - name:         构件在 FBX 里的原始名（CG-001 / B_JZ_006 等）
    //
    // ⚠️ Batch Table JSON 只放 string / bool / array 类型，不能放 numeric。
    //    Cesium 1.95+ 用这个格式让 `Cesium3DTileFeature.getProperty('name')` 工作。
    let names: Vec<String> = geometry.batch_names.clone();
    // id 与 scenetree.json 同算法：FNV-1a(name#g全局序号)——同一构件跨 tile id 一致，
    // 页面可以用 pick 到的 id 直接在 scenetree 里定位/高亮。
    let gids = &geometry.batch_gids;
    let ids: Vec<String> = (0..names.len()).map(|i| {
        let gid = gids.get(i).copied().unwrap_or(i);
        let s = format!("{}#g{}", names.get(i).cloned().unwrap_or_default(), gid);
        // 简单 FNV-1a 32-bit hex
        let mut h: u32 = 0x811c9dc5;
        for b in s.bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(0x01000193);
        }
        format!("{:08x}{:08x}{:08x}{:08x}", h, h.wrapping_add(gid as u32), 0u32, 0u32)
    }).collect();
    let pack_props: Vec<&str> = vec!["null"; names.len().max(1)];
    let bt = json!({
        "_pack_props_": pack_props,
        "id": ids,
        "name": names,
    });
    let bt_str = serde_json::to_string(&bt)?;
    let bt_bytes = bt_str.as_bytes();
    let bt_pad = (8 - (bt_bytes.len() % 8)) % 8;
    let bt_chunk_len = bt_bytes.len() + bt_pad;

    // b3dm header 28B
    let b3dm_len = 28 + ft_chunk_len + bt_chunk_len + glb.len();

    let mut out = Vec::with_capacity(b3dm_len);
    // magic
    out.extend_from_slice(b"b3dm");
    out.extend_from_slice(&1u32.to_le_bytes()); // version
    out.extend_from_slice(&(b3dm_len as u32).to_le_bytes()); // byteLength
    out.extend_from_slice(&(ft_chunk_len as u32).to_le_bytes()); // featureTableJSONByteLength
    out.extend_from_slice(&0u32.to_le_bytes()); // featureTableBINByteLength
    out.extend_from_slice(&(bt_chunk_len as u32).to_le_bytes()); // batchTableJSONByteLength
    out.extend_from_slice(&0u32.to_le_bytes()); // batchTableBINByteLength
    // Feature Table
    out.extend_from_slice(ft_bytes);
    out.extend(std::iter::repeat(b' ').take(ft_pad));
    // Batch Table
    out.extend_from_slice(bt_bytes);
    out.extend(std::iter::repeat(b' ').take(bt_pad));
    // GLB payload
    out.extend_from_slice(&glb);

    Ok(out)
}

/// 验证 JSON（用于 debug 输出，不影响构建）
#[allow(dead_code)]
pub fn _validate_json(v: &Value) -> Result<()> {
    serde_json::to_string(v).context("JSON 校验")?;
    Ok(())
}