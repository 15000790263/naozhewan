//! _3dtile - FBX → 3D Tiles 转换工具（含 LOD 支持）
//!
//! 当前版本：B0 八叉树切块阶段（仅切块 + dump 验证，不写 b3dm）

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

mod octree;
mod b3dm;

use octree::{Aabb, OctreeBuilder};
use b3dm::{build_b3dm, merge_meshes_by_texture, simplify_mesh, MeshData, TileGeometry};

#[derive(Parser, Debug)]
#[command(name = "_3dtile", about = "FBX → 3D Tiles 转换工具", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// 探针：解析 FBX，输出节点/mesh/材质/贴图统计
    Probe {
        /// 输入 FBX 文件路径
        input: PathBuf,
    },

    /// 贴图诊断：dump 每个 mesh 的全部 material 和 baseColor 贴图链
    /// （检查 FBX 里 mesh.material 是否指错贴图，用于排查"全 mesh 用同一张图"问题）
    DumpMat {
        /// 输入 FBX 文件路径
        input: PathBuf,
    },

    /// 切片：用八叉树把 FBX 切成 hierarchical tile，输出切块统计
    ///
    /// 当前只切块不写 b3dm。验证切块粒度合理后，再进入 B1（写 b3dm）。
    Slice {
        /// 输入 FBX 文件路径
        input: PathBuf,

        /// 八叉树最大深度（默认 4，深度 d 共 8^d 个潜在叶节点）
        #[arg(long, default_value_t = 4)]
        max_depth: u32,

        /// 每叶节点最多 mesh 数（默认 200）
        #[arg(long, default_value_t = 200)]
        max_meshes: usize,
    },

    /// 构建：FBX → 八叉树切块 → 写 b3dm + tileset.json
    ///
    /// 输出到指定目录，结构：
    ///   <output>/
    ///     tileset.json
    ///     tiles/tile_0.b3dm, tile_1.b3dm, ...
    Build {
        /// 输入 FBX 文件路径
        input: PathBuf,

        /// 输出目录（不存在会自动创建）
        output: PathBuf,

        /// 八叉树最大深度（默认 6）。体积预算（--tile-size-mb）会自动让 tile 停在
        /// ~10MB，深度只是硬上限，防止个别空间挤满大 mesh 时切不动。
        #[arg(long, default_value_t = 6)]
        max_depth: u32,

        /// 每叶节点最多 mesh 数
        #[arg(long, default_value_t = 200)]
        max_meshes: usize,

        /// 目标单 tile 大小（MB，默认 20 = 用户允许范围的上限）。
        /// 切分按"三角形数 × 63B"预算控制。默认偏上限是因为 tile 数少 → Cesium
        /// 在 frustum 边界 evict/重载 tile 频率低 → 拖动流畅（实测对比：
        /// tile_size_mb=10 → 79 tile 卡顿；tile_size_mb=20 → ~40 tile 接近
        /// CesiumLab mainview 26 tile 的流畅度）。
        /// 改小→tile 更多更细（单 tile 精度更高）；改大→tile 更少更粗。
        #[arg(long, default_value_t = 20.0)]
        tile_size_mb: f64,

        /// LOD 层数：0 = 单层（只有最细层，默认）；N = 生成 N+1 层金字塔
        ///
        /// 层级越粗，用顶点聚类简化把顶点按更大网格合并，三角形数大幅下降。
        /// 远景只加载粗层，实现真正的按距离切换。
        #[arg(long, default_value_t = 0)]
        lod: u32,

        /// 模型定位经度（度）。不传则使用北京天安门默认值。
        #[arg(long, default_value_t = 116.397428)]
        longitude: f64,

        /// 模型定位纬度（度）。不传则使用北京天安门默认值。
        #[arg(long, default_value_t = 39.90923)]
        latitude: f64,

        /// 模型定向角度 heading（度，0=正北，-178=湛江油库使用的近似南偏东方向）。默认 0。
        #[arg(long, default_value_t = 0.0)]
        heading: f64,

        /// 模型定向角度 pitch（度）。默认 0（水平摆放，Cesium 自动适应 up 轴）。
        #[arg(long, default_value_t = 0.0)]
        pitch: f64,

        /// 模型定向角度 roll（度）。默认 0。
        #[arg(long, default_value_t = 0.0)]
        roll: f64,

        /// 贴地微调（米，负=下沉）。自动贴地基于顶点 Y 的 5% 分位数，
        /// 若模型仍有悬空/下沉，用此参数微调，如 --ground-offset=-1.5 再下沉 1.5m。
        #[arg(long, default_value_t = 0.0)]
        ground_offset: f64,

        /// 材质粗糙度（0=镜面，1=全漫反射）。默认 0.45 对齐 CesiumLab 输出。
        ///
        /// 【为什么默认是 0.45 而不是读 FBX 原值】实测 ABC831.fbx 自带 roughness=0.859
        /// （接近全漫反射）→ 模型表面没有环境高光/反射，观感发闷；CesiumLab mainview
        /// 输出固定 roughness=0.45 → 表面有明显环境反射。默认覆盖 FBX 值以对齐观感，
        /// 想忠实还原 FBX 材质时传 `--roughness 0.859`。
        #[arg(long, default_value_t = 0.45)]
        roughness: f64,

        /// 材质金属度（0=非金属，1=纯金属）。默认 0.0（与 glTF/FBX 默认一致）。
        #[arg(long, default_value_t = 0.0)]
        metallic: f64,

        /// 原点平移：沿模型局部 X 轴平移（米，默认 0）。
        /// 用于把不同 FBX 导出的同一场景对齐到相同坐标系——同一片区域的两个 FBX
        /// 常因为导出原点不同而有固定偏移（实测 ABC831 与未命名.fbx 差 70.9m），
        /// 之前基于其中一个拾取的经纬度坐标（鹤位、区域）套到另一个上就会整体偏。
        #[arg(long, default_value_t = 0.0)]
        origin_offset_x: f64,

        /// 原点平移：沿模型局部 Z 轴平移（米，默认 0）。用法同 --origin-offset-x。
        #[arg(long, default_value_t = 0.0)]
        origin_offset_z: f64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Probe { input } => probe(&input),
        Cmd::DumpMat { input } => dump_mat(&input),
        Cmd::Slice {
            input,
            max_depth,
            max_meshes,
        } => slice(&input, max_depth, max_meshes),
        Cmd::Build {
            input,
            output,
            max_depth,
            max_meshes,
            tile_size_mb,
            lod,
            longitude,
            latitude,
            heading,
            pitch,
            roll,
            ground_offset,
            roughness,
            metallic,
            origin_offset_x,
            origin_offset_z,
        } => build_cmd(
            &input,
            &output,
            max_depth,
            max_meshes,
            tile_size_mb,
            lod,
            longitude,
            latitude,
            heading,
            pitch,
            roll,
            ground_offset,
            roughness,
            metallic,
            origin_offset_x,
            origin_offset_z,
        ),
    }
}

/// FBX 探针：解析并输出统计信息
fn probe(input: &PathBuf) -> Result<()> {
    let meta = std::fs::metadata(input)
        .with_context(|| format!("无法读取文件元数据: {}", input.display()))?;
    let size_mb = meta.len() as f64 / 1024.0 / 1024.0;

    eprintln!("[probe] 输入文件: {}", input.display());
    eprintln!("[probe] 文件大小: {:.2} MiB", size_mb);
    eprintln!("[probe] ufbx 解析中...");

    let start = Instant::now();
    let path_str = input
        .to_str()
        .context("路径不是合法 UTF-8，无法传给 ufbx")?;
    let scene = ufbx::load_file(path_str, ufbx::LoadOpts::default())
        .map_err(|e| anyhow::anyhow!("ufbx 解析失败: {} - {:?}", input.display(), e))?;
    let elapsed = start.elapsed();

    eprintln!("[probe] 解析完成，耗时: {:.2?}", elapsed);

    let mesh_count = scene.meshes.len();
    let mat_count = scene.materials.len();
    let tex_count = scene.textures.len();
    let node_count = scene.nodes.len();

    let total_verts: usize = scene.meshes.iter().map(|m| m.vertices.len()).sum();
    let total_tris: usize = scene.meshes.iter().map(|m| m.faces.len()).sum();

    println!("\n========== FBX 探针报告 ==========");
    println!("节点总数         : {}", node_count);
    println!("Mesh 数          : {}", mesh_count);
    println!("材质数           : {}", mat_count);
    println!("贴图数           : {}", tex_count);
    println!("总顶点数         : {}", total_verts);
    println!("总三角形数       : {}", total_tris);
    println!("==================================\n");

    println!("前 10 个 mesh（用于判断切块粒度）:");
    for (i, m) in scene.meshes.iter().take(10).enumerate() {
        println!(
            "  [{:>2}] 顶点数={:<8} 三角形={:<8} 名称={}",
            i,
            m.vertices.len(),
            m.faces.len(),
            m.element.name
        );
    }
    if mesh_count > 10 {
        println!("  ... 还有 {} 个 mesh", mesh_count - 10);
    }
    println!();

    println!("贴图列表（前 10）:");
    for (i, t) in scene.textures.iter().take(10).enumerate() {
        println!(
            "  [{:>2}] 名称={} 文件={:?}",
            i,
            t.element.name,
            t.filename
        );
    }
    if tex_count > 10 {
        println!("  ... 还有 {} 张贴图", tex_count - 10);
    }

    Ok(())
}

/// dump 每个 mesh 的全部 material 和 baseColor 贴图链
fn dump_mat(input: &PathBuf) -> Result<()> {
    let path_str = input.to_str().context("路径不是合法 UTF-8")?;
    let scene = ufbx::load_file(path_str, ufbx::LoadOpts::default())
        .map_err(|e| anyhow::anyhow!("ufbx 解析失败: {} - {:?}", input.display(), e))?;

    eprintln!("=== TOTAL ===");
    eprintln!("  meshes={}  materials={}  textures={}",
        scene.meshes.len(), scene.materials.len(), scene.textures.len());

    // mesh.materials 数量分布
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for m in &scene.meshes {
        *counts.entry(m.materials.len()).or_insert(0) += 1;
    }
    eprintln!("\n=== mesh.materials 数量分布 ===");
    for (k, v) in counts.iter() {
        eprintln!("  mats_per_mesh={}: {} 个 mesh", k, v);
    }

    // 【贴图错乱诊断】face_material 完整性：长度是否等于 faces.len()、索引是否越界
    let mut fm_ok = 0;
    let mut fm_short = 0;   // 长度 < faces.len()
    let mut fm_empty = 0;   // 长度 0
    let mut fm_oob = 0;     // 索引 >= materials.len()
    for m in &scene.meshes {
        if m.face_material.is_empty() {
            fm_empty += 1;
        } else if m.face_material.len() != m.faces.len() {
            fm_short += 1;
        } else {
            fm_ok += 1;
            let max_idx = m.face_material.iter().max().copied().unwrap_or(0) as usize;
            if max_idx >= m.materials.len() {
                fm_oob += 1;
            }
        }
    }
    eprintln!("\n=== face_material 完整性（贴图错乱诊断）===");
    eprintln!("  完整(长度==faces且索引不越界): {} | 空(整个mesh回退materials[0]): {} | 长度不等(部分面回退?): {} | 索引越界: {}",
        fm_ok, fm_empty, fm_short, fm_oob);

    // 抽查多材质 mesh 的 face 归属：materials.len()>1 的 mesh 里，face_material 实际用到几个
    let mut multi_full = 0;
    let mut multi_flat = 0; // face_material 全为同一值（实际没有按面区分材质）
    let mut multi_sample: Vec<String> = Vec::new();
    for m in &scene.meshes {
        if m.materials.len() > 1 && m.face_material.len() == m.faces.len() && !m.faces.is_empty() {
            let uniq: std::collections::HashSet<u32> = m.face_material.iter().copied().collect();
            if uniq.len() == 1 {
                multi_flat += 1;
            } else {
                multi_full += 1;
                if multi_sample.len() < 3 {
                    multi_sample.push(format!("{}: {} mats 实际用 {} 组", m.element.name, m.materials.len(), uniq.len()));
                }
            }
        }
    }
    eprintln!("  多材质mesh且face_material有效: 实际多组 {} | 其实全同一组(等效单材质) {}", multi_full, multi_flat);
    for s in &multi_sample {
        eprintln!("    例: {}", s);
    }

    // 【属性探查】node 用户属性：FBX 自定义参数（设备参数/规格表）常挂 node 层
    // API：node.element.props.props = List<Prop>；Prop 有 name/value_str/value_int/value_vec4/type_
    let fmt_prop = |p: &ufbx::Prop| -> String {
        match p.type_ {
            ufbx::PropType::String => format!("{}='{}'", p.name, p.value_str),
            ufbx::PropType::Integer | ufbx::PropType::Boolean => format!("{}={}", p.name, p.value_int),
            ufbx::PropType::Number => format!("{}={:.4}", p.name, p.value_vec4.x),
            _ => format!("{}=({:?})", p.name, p.type_),
        }
    };
    let mut with_props = 0;
    let mut prop_keys: HashMap<String, usize> = HashMap::new();
    let mut samples: Vec<String> = Vec::new();
    for node in &scene.nodes {
        let count = node.element.props.props.len();
        if count == 0 {
            continue;
        }
        with_props += 1;
        for p in node.element.props.props.iter() {
            *prop_keys.entry(p.name.to_string()).or_insert(0) += 1;
        }
        if samples.len() < 3 {
            let kv: Vec<String> = node.element.props.props.iter().take(8)
                .map(&fmt_prop).collect();
            samples.push(format!("node '{}': {}", node.element.name, kv.join(", ")));
        }
    }
    eprintln!("\n=== node 用户属性探查 ===");
    eprintln!("  有属性的 node: {} / {}", with_props, scene.nodes.len());
    let mut keys: Vec<_> = prop_keys.keys().cloned().collect();
    keys.sort();
    for k in keys {
        eprintln!("    {}: {} 次", k, prop_keys[&k]);
    }
    for s in &samples {
        eprintln!("  样例: {}", s);
    }

    // UDP3DSMAX（3ds Max 用户自定义属性）内容样例——业务属性通常在这
    let mut udp_shown = 0;
    for node in &scene.nodes {
        if udp_shown >= 5 {
            break;
        }
        if let Some(p) = node.element.props.find_prop("UDP3DSMAX") {
            if !p.value_str.is_empty() {
                eprintln!("  UDP3DSMAX [{}] = {:?}", node.element.name, p.value_str);
                udp_shown += 1;
            }
        }
    }
    if udp_shown == 0 {
        eprintln!("  （UDP3DSMAX 全为空）");
    }

    // 全 material 名 + baseColor 贴图 + PBR 参数
    eprintln!("\n=== 所有 material (按 baseColor 贴图 + PBR) ===");
    let mut by_tex: HashMap<String, Vec<String>> = HashMap::new();
    for mat in &scene.materials {
        let mat_name = mat.element.name.to_string();
        let tex_file = mat.pbr.base_color.texture.as_ref()
            .map(|t| t.filename.to_string())
            .unwrap_or_else(|| "(无贴图)".to_string());
        let metal = mat.pbr.metalness.has_value;
        let metal_x = mat.pbr.metalness.value_vec4.x;
        let rough = mat.pbr.roughness.has_value;
        let rough_x = mat.pbr.roughness.value_vec4.x;
        let key = format!("{}|m={}{:.3}|r={}{:.3}", tex_file,
            if metal {"✓"} else {"✗"}, metal_x,
            if rough {"✓"} else {"✗"}, rough_x);
        by_tex.entry(key).or_default().push(mat_name);
    }
    let mut keys: Vec<_> = by_tex.keys().collect();
    keys.sort();
    for tex in keys {
        let mats = &by_tex[tex];
        eprintln!("  [{}] {}  → mats: {:?}", mats.len(), tex,
            mats.iter().take(2).cloned().collect::<Vec<_>>());
    }

    // 多 material 的 mesh 抽样
    let multi: Vec<_> = scene.meshes.iter()
        .enumerate()
        .filter(|(_, m)| m.materials.len() > 1)
        .take(8)
        .collect();
    eprintln!("\n=== 多 material mesh 抽样 ===");
    for (i, m) in &multi {
        eprintln!("  [{}] {} ({} mats)", i, m.element.name, m.materials.len());
        for (mi, mat) in m.materials.iter().enumerate() {
            let tex_file = mat.pbr.base_color.texture.as_ref()
                .map(|t| t.filename.to_string())
                .unwrap_or_else(|| "(无贴图)".to_string());
            eprintln!("    [mat{}] name={:?} tex={}", mi, mat.element.name.to_string(), tex_file);
        }
    }
    let total_multi = scene.meshes.iter().filter(|m| m.materials.len() > 1).count();
    eprintln!("\n  总共 {} 个多 material mesh（占 {:.1}%）",
        total_multi, total_multi as f64 / scene.meshes.len() as f64 * 100.0);

    Ok(())
}

/// 八叉树切块：解析 FBX + 计算每个 mesh 中心点 + 构建八叉树 + 输出统计
fn slice(input: &PathBuf, max_depth: u32, max_meshes: usize) -> Result<()> {
    let meta = std::fs::metadata(input)
        .with_context(|| format!("无法读取文件元数据: {}", input.display()))?;
    eprintln!("[slice] 输入: {} ({:.2} MiB)", input.display(), meta.len() as f64 / 1024.0 / 1024.0);
    eprintln!("[slice] 八叉树参数: max_depth={} max_meshes/leaf={}", max_depth, max_meshes);

    // 1. 解析 FBX
    let start = Instant::now();
    let path_str = input.to_str().context("路径不是合法 UTF-8")?;
    let scene = ufbx::load_file(path_str, ufbx::LoadOpts::default())
        .map_err(|e| anyhow::anyhow!("ufbx 解析失败: {} - {:?}", input.display(), e))?;
    eprintln!("[slice] ufbx 解析完成: {} 个 mesh，耗时 {:.2?}", scene.meshes.len(), start.elapsed());

    // 2. 计算每个 mesh 的中心点（从 vertices 求平均）
    let t1 = Instant::now();
    let mut root_bounds = Aabb::empty();
    let mesh_centers: Vec<[f64; 3]> = scene
        .meshes
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let n = m.num_vertices as usize;
            if n == 0 {
                eprintln!("[slice] 警告: mesh {} ({}) 无顶点", i, m.element.name);
                return [0.0, 0.0, 0.0];
            }
            let mut sum_x = 0.0_f64;
            let mut sum_y = 0.0_f64;
            let mut sum_z = 0.0_f64;
            for v in &m.vertices {
                // ufbx 0.11 的 vertices 元素类型是 Vec3
                sum_x += v.x as f64;
                sum_y += v.y as f64;
                sum_z += v.z as f64;
            }
            let center = [sum_x / n as f64, sum_y / n as f64, sum_z / n as f64];
            root_bounds.expand_point(center);
            center
        })
        .collect();

    // 给 root AABB 加点 padding，避免边界 mesh 落不到任何子节点
    let pad = root_bounds.max_extent() * 0.001;
    for i in 0..3 {
        root_bounds.min[i] -= pad;
        root_bounds.max[i] += pad;
    }

    eprintln!(
        "[slice] 根 AABB: min=({:.2}, {:.2}, {:.2}) max=({:.2}, {:.2}, {:.2}) 耗时 {:.2?}",
        root_bounds.min[0], root_bounds.min[1], root_bounds.min[2],
        root_bounds.max[0], root_bounds.max[1], root_bounds.max[2],
        t1.elapsed()
    );

    // 3. 构建八叉树
    let t2 = Instant::now();
    let mesh_tri_counts: Vec<u64> = scene.meshes.iter().map(|m| m.num_triangles as u64).collect();
    let max_tri_per_tile = (10.0f64 * 1_000_000.0 / 63.0).max(1.0) as u64; // 目标 ~10MB/tile
    let builder = OctreeBuilder {
        max_depth,
        max_meshes_per_leaf: max_meshes,
        min_extent: 0.5, // 包围盒任一轴 < 0.5m 停止细分
        max_triangles_per_tile: max_tri_per_tile,
    };
    // root_bounds 会被 build() 消费，clone 一份用于打印报告
    let root_bounds_for_report = root_bounds.clone();
    let tree = builder.build(root_bounds, &mesh_centers, &mesh_tri_counts);
    eprintln!("[slice] 八叉树构建完成，耗时 {:.2?}", t2.elapsed());

    // 4. 输出统计
    let total_leaves = tree.leaf_count();
    let total_mesh_in_tree = tree.total_mesh_count();
    let mut depth_count: HashMap<u32, usize> = HashMap::new();
    tree.depth_histogram(&mut depth_count);

    println!("\n========== 八叉树切块报告 ==========");
    println!("根 AABB 范围: {:.2} x {:.2} x {:.2} 米",
        root_bounds_for_report.max[0] - root_bounds_for_report.min[0],
        root_bounds_for_report.max[1] - root_bounds_for_report.min[1],
        root_bounds_for_report.max[2] - root_bounds_for_report.min[2]);
    println!("叶节点数: {}", total_leaves);
    println!("覆盖 mesh 总数: {} (输入 {}，{})",
        total_mesh_in_tree,
        scene.meshes.len(),
        if total_mesh_in_tree == scene.meshes.len() { "✓ 一致" } else { "✗ 不一致！" });
    println!("深度分布:");
    let mut depths: Vec<u32> = depth_count.keys().copied().collect();
    depths.sort();
    for d in depths {
        println!("  深度 {}: {} 个节点", d, depth_count[&d]);
    }
    println!("=====================================\n");

    // 5. 输出叶节点 mesh 数分布（评估切块均匀度）
    let mut leaves = Vec::new();
    tree.collect_leaves(&mut leaves);
    leaves.sort_by(|a, b| b.mesh_ids.len().cmp(&a.mesh_ids.len()));

    let mut mesh_counts: Vec<usize> = leaves.iter().map(|l| l.mesh_ids.len()).collect();
    mesh_counts.sort();
    let median = if mesh_counts.is_empty() {
        0
    } else {
        mesh_counts[mesh_counts.len() / 2]
    };
    let max_mesh = mesh_counts.last().copied().unwrap_or(0);
    let min_mesh = mesh_counts.first().copied().unwrap_or(0);

    println!("叶节点 mesh 数分布:");
    println!("  min={} median={} max={}", min_mesh, median, max_mesh);
    println!();

    println!("Top 10 mesh 数最多的叶节点:");
    for (i, leaf) in leaves.iter().take(10).enumerate() {
        let tris: usize = leaf.mesh_ids.iter()
            .map(|&id| scene.meshes[id].faces.len())
            .sum();
        println!(
            "  [{:>2}] mesh数={:<5} 三角形数={:<8} 深度={} 中心=({:.2}, {:.2}, {:.2})",
            i,
            leaf.mesh_ids.len(),
            tris,
            leaf.depth,
            leaf.bounds.center()[0],
            leaf.bounds.center()[1],
            leaf.bounds.center()[2],
        );
    }
    println!();

    println!("Bottom 10 mesh 数最少的叶节点:");
    for (i, leaf) in leaves.iter().rev().take(10).enumerate() {
        println!(
            "  [{:>2}] mesh数={} 深度={} 中心=({:.2}, {:.2}, {:.2})",
            i,
            leaf.mesh_ids.len(),
            leaf.depth,
            leaf.bounds.center()[0],
            leaf.bounds.center()[1],
            leaf.bounds.center()[2],
        );
    }
    println!();

    // 6. 空叶节点统计（包围盒内没有 mesh 的格子）
    let empty_leaves = leaves.iter().filter(|l| l.mesh_ids.is_empty()).count();
    println!("空叶节点数: {} / {} ({:.1}%)",
        empty_leaves, total_leaves,
        empty_leaves as f64 / total_leaves as f64 * 100.0);
    if empty_leaves > 0 {
        println!("提示：空叶节点会在 B1 阶段过滤掉，不生成 b3dm。");
    }

    Ok(())
}

/// Build 命令：FBX → 八叉树切块 → b3dm + tileset.json
fn build_cmd(
    input: &PathBuf,
    output: &PathBuf,
    max_depth: u32,
    max_meshes: usize,
    tile_size_mb: f64,
    lod: u32,
    longitude: f64,
    latitude: f64,
    heading: f64,
    pitch: f64,
    roll: f64,
    ground_offset: f64,
    roughness: f64,
    metallic: f64,
    origin_offset_x: f64,
    origin_offset_z: f64,
) -> Result<()> {
    eprintln!("[build] 输入: {}", input.display());
    eprintln!(
        "[build] 定位: 经度={:.5}° 纬度={:.5}° heading={:.2}° pitch={:.2}° roll={:.2}°",
        longitude, latitude, heading, pitch, roll
    );
    eprintln!("[build] 输出目录: {}", output.display());
    eprintln!(
        "[build] 八叉树参数: max_depth={} max_meshes/leaf={} 目标tile={:.1}MB",
        max_depth, max_meshes, tile_size_mb
    );
    eprintln!(
        "[build] 材质: metallic={:.2} roughness={:.2}（覆盖 FBX 原值以对齐 CesiumLab 观感）",
        metallic, roughness
    );

    // 创建输出目录
    std::fs::create_dir_all(output).with_context(|| format!("创建输出目录失败: {}", output.display()))?;
    // b3dm 直接放在 output 根目录（与 tileset.json 同级），对齐 CesiumLab 输出结构
    // 贴图以内嵌 data URI 写入 b3dm 的 GLB，不再生成外置 textures/ 目录。

    // FBX 输入所在目录（贴图通常在 .fbm 子目录里，和 FBX 同级）
    let input_dir = input.parent().context("FBX 路径无父目录")?;

    // 1. 解析 FBX（用 LoadOpts 转轴 + 单位换算，让 FBX 默认 Z-up/cm 转成 Y-up/米）
    let path_str = input.to_str().context("路径不是合法 UTF-8")?;
    let load_opts = ufbx::LoadOpts::default();
    let scene = ufbx::load_file(path_str, load_opts)
        .map_err(|e| anyhow::anyhow!("ufbx 解析失败: {} - {:?}", input.display(), e))?;
    eprintln!("[build] ufbx 解析完成: {} 个 mesh", scene.meshes.len());

    // 2. 【关键修复】收集 mesh -> node.geometry_to_world 变换矩阵
    // ufbx 的 mesh 顶点在 **geometry 局部空间**，必须乘 node 的 geometry_to_world
    // 才是场景坐标。之前一直没用——大多数 mesh 的节点变换恰好互补所以"整体看起来对"，
    // 树类节点带个体旋转/镜像就暴露成"树根朝上"。
    // g2w 输出是 FBX 原生 **厘米 + Y-up**：需要 scale 0.01 转米。
    let mut mesh_to_world: HashMap<usize, ufbx::Matrix> = HashMap::new();
    // mesh 指针 → node 名：ABC831 这类导出把名字/属性挂在 node 层，mesh.element.name 反而为空。
    // pick 属性（Batch Table / scenetree）的构件名 fallback 链 = mesh 名 → node 名 → mesh_<gid>
    let mut mesh_node_names: HashMap<usize, String> = HashMap::new();
    for node in &scene.nodes {
        if let Some(mesh_ref) = &node.mesh {
            let mesh: &ufbx::Mesh = mesh_ref;
            mesh_to_world.insert(mesh as *const ufbx::Mesh as usize, node.geometry_to_world);
            let nn = node.element.name.to_string();
            if !nn.is_empty() {
                mesh_node_names.entry(mesh as *const ufbx::Mesh as usize).or_insert(nn);
            }
        }
    }

    // 2b. 第一遍：全模型 AABB（应用 g2w 后，cm、Y-up）——用于居中
    let mut gmin = [f64::INFINITY; 3];
    let mut gmax = [f64::NEG_INFINITY; 3];
    // 同时收集每个 mesh 变换后的最低 Y，用于稳健地确定"地面高度"。
    // 直接取全局最小 Y 会被单个异常低点带偏（实测 4111 个 primitive 里
    // 只有 1 个在 Y=0，其余都 ≥5m，真实地面在 Y≈8.5m）。
    let mut mesh_min_y: Vec<f64> = Vec::new();
    for node in &scene.nodes {
        if let Some(mesh_ref) = &node.mesh {
            let mesh: &ufbx::Mesh = mesh_ref;
            let m = node.geometry_to_world;
            let cnt = mesh.vertex_position.values.count;
            if cnt == 0 {
                continue;
            }
            let base = mesh.vertex_position.values.data;
            let mut mn = [f64::INFINITY; 3];
            let mut mx = [f64::NEG_INFINITY; 3];
            for i in 0..cnt {
                let p = unsafe { base.add(i).read() };
                let pv = [p.x, p.y, p.z];
                for k in 0..3 {
                    if pv[k] < mn[k] { mn[k] = pv[k]; }
                    if pv[k] > mx[k] { mx[k] = pv[k]; }
                }
            }
            // 8 角点经 g2w 变换取包络（含旋转时保守正确）
            for c in 0..8usize {
                let p = ufbx::Vec3 {
                    x: if c & 1 != 0 { mx[0] } else { mn[0] },
                    y: if c & 2 != 0 { mx[1] } else { mn[1] },
                    z: if c & 4 != 0 { mx[2] } else { mn[2] },
                };
                let w = ufbx::transform_position(&m, p);
                let wv = [w.x, w.y, w.z];
                for k in 0..3 {
                    if wv[k] < gmin[k] { gmin[k] = wv[k]; }
                    if wv[k] > gmax[k] { gmax[k] = wv[k]; }
                }
            }
            // 记录该 mesh 变换后的最低 Y（wmn[1] 需要单独算，这里用 8 角点的 min）
            let mut mesh_lo_y = f64::INFINITY;
            for c in 0..8usize {
                let p = ufbx::Vec3 {
                    x: if c & 1 != 0 { mx[0] } else { mn[0] },
                    y: if c & 2 != 0 { mx[1] } else { mn[1] },
                    z: if c & 4 != 0 { mx[2] } else { mn[2] },
                };
                let w = ufbx::transform_position(&m, p);
                if w.y < mesh_lo_y { mesh_lo_y = w.y; }
            }
            if mesh_lo_y.is_finite() {
                mesh_min_y.push(mesh_lo_y);
            }
        }
    }

    // 【单位自适应】用 FBX 自带的全局单位设置（1 个 FBX 单位 = 多少米），
    // 不再硬编码 0.01（假设厘米）。硬编码的坑：米单位的 FBX（如 ABC831）会被
    // 整体缩小 100 倍——实测 boundingSphere 半径只剩 4.47m（正确应 ~447m），
    // 整个模型缩成地面上的一个小点。取不到有效值时才退回厘米假设。
    let unit_meters = scene.settings.unit_meters;
    let model_scale = if unit_meters.is_finite() && unit_meters > 0.0 {
        unit_meters
    } else {
        0.01
    };
    eprintln!(
        "[build] FBX 单位: 1 单位 = {} 米 → scale={}（{}）",
        unit_meters,
        model_scale,
        if (model_scale - 0.01).abs() < 1e-9 { "厘米单位" } else { "非厘米单位，已自适应" }
    );
    // 地面高度 = 各 mesh 最低 Y 的 5% 分位数。
    // 用分位数而不是全局最小值，避免个别异常低点（如地下基础、
    // 孤立碎片）把整个模型抬起来造成"悬空"。
    // --ground-offset：手动微调（米，负=下沉），自动分位数不完美时用。
    let ground_y = if mesh_min_y.is_empty() {
        gmin[1]
    } else {
        mesh_min_y.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((mesh_min_y.len() as f64) * 0.05) as usize;
        mesh_min_y[idx.min(mesh_min_y.len() - 1)]
    };
    let ground_y = ground_y + ground_offset / model_scale; // offset 米 → 厘米

    // 162MB 版本：model_off[1] = ground_y * scale（不翻转）
    let model_off = [
        origin_offset_x,
        ground_y * model_scale,
        origin_offset_z,
    ];
    eprintln!(
        "[build] 地面高度: 全局最低 Y={:.2}m, 5%分位={:.2}m, offset={:.2}m → 采用 {:.2}m 作为贴地基准",
        gmin[1] * model_scale,
        (ground_y - ground_offset / model_scale) * model_scale,
        ground_offset,
        ground_y * model_scale,
    );

    // 临时诊断：dump 第一个含 PNG（alpha）贴图的 mesh 的 g2w 矩阵与前后 bbox
    if std::env::var("DIAG_TREE").is_ok() {
        for node in &scene.nodes {
            if let Some(mesh_ref) = &node.mesh {
                let mesh: &ufbx::Mesh = mesh_ref;
                let has_png = mesh.materials.iter().any(|mat| {
                    mat.pbr.base_color.texture.as_ref()
                        .map(|t| t.filename.to_string().to_lowercase().ends_with(".png"))
                        .unwrap_or(false)
                });
                if !has_png || mesh.num_vertices == 0 {
                    continue;
                }
                let m = node.geometry_to_world;
                eprintln!("[diag-tree] mesh={} 原始顶点数={}", mesh.element.name, mesh.num_vertices);
                // 原始 bbox
                let mut mn = [f64::INFINITY; 3];
                let mut mx = [f64::NEG_INFINITY; 3];
                for i in 0..mesh.num_vertices as usize {
                    let p = unsafe { mesh.vertex_position.values.data.add(i).read() };
                    let pv = [p.x, p.y, p.z];
                    for k in 0..3 {
                        if pv[k] < mn[k] { mn[k] = pv[k]; }
                        if pv[k] > mx[k] { mx[k] = pv[k]; }
                    }
                }
                eprintln!("[diag-tree] 原始 bbox(cm): X[{:.1},{:.1}] Y[{:.1},{:.1}] Z[{:.1},{:.1}]", mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]);
                eprintln!("[diag-tree] g2w 矩阵:");
                eprintln!("  | {:8.3} {:8.3} {:8.3} {:10.2} |", m.m00, m.m01, m.m02, m.m03);
                eprintln!("  | {:8.3} {:8.3} {:8.3} {:10.2} |", m.m10, m.m11, m.m12, m.m13);
                eprintln!("  | {:8.3} {:8.3} {:8.3} {:10.2} |", m.m20, m.m21, m.m22, m.m23);
                // 变换后 bbox（8 角点）
                let mut wmn = [f64::INFINITY; 3];
                let mut wmx = [f64::NEG_INFINITY; 3];
                for c in 0..8usize {
                    let p = ufbx::Vec3 {
                        x: if c & 1 != 0 { mx[0] } else { mn[0] },
                        y: if c & 2 != 0 { mx[1] } else { mn[1] },
                        z: if c & 4 != 0 { mx[2] } else { mn[2] },
                    };
                    let w = ufbx::transform_position(&m, p);
                    let wv = [w.x, w.y, w.z];
                    for k in 0..3 {
                        if wv[k] < wmn[k] { wmn[k] = wv[k]; }
                        if wv[k] > wmx[k] { wmx[k] = wv[k]; }
                    }
                }
                eprintln!("[diag-tree] 变换后 bbox(cm): X[{:.1},{:.1}] Y[{:.1},{:.1}] Z[{:.1},{:.1}]", wmn[0], wmx[0], wmn[1], wmx[1], wmn[2], wmx[2]);
                break;
            }
        }
    }

    // 3. 计算 mesh 中心点（geometry 空间中心经 g2w + scale + off 变换到输出坐标）+ 构建八叉树
    let mut root_bounds = Aabb::empty();
    let mesh_centers: Vec<[f64; 3]> = scene
        .meshes
        .iter()
        .map(|m| {
            let n = m.num_vertices as usize;
            if n == 0 {
                return [0.0, 0.0, 0.0];
            }
            let mut sum_x = 0.0_f64;
            let mut sum_y = 0.0_f64;
            let mut sum_z = 0.0_f64;
            for v in &m.vertices {
                sum_x += v.x as f64;
                sum_y += v.y as f64;
                sum_z += v.z as f64;
            }
            let c0 = [sum_x / n as f64, sum_y / n as f64, sum_z / n as f64];
            // 应用 geometry_to_world + cm→m + 居中偏移
            let mesh_ref: &ufbx::Mesh = m;
            let key = mesh_ref as *const ufbx::Mesh as usize;
            let center = match mesh_to_world.get(&key) {
                Some(mat) => {
                    let w = ufbx::transform_position(mat, ufbx::Vec3 { x: c0[0], y: c0[1], z: c0[2] });
                    // 162MB 版本：直接 (x, y, z) 出（不加翻转）
                    [w.x * model_scale - model_off[0],
                     w.y * model_scale - model_off[1],
                     w.z * model_scale - model_off[2]]
                }
                None => c0,
            };
            root_bounds.expand_point(center);
            center
        })
        .collect();

    // padding
    let pad = root_bounds.max_extent() * 0.001;
    for i in 0..3 {
        root_bounds.min[i] -= pad;
        root_bounds.max[i] += pad;
    }

    // mesh 三角形数（fan 后近似取 ufbx num_triangles）：体积预算切分用
    let mesh_tri_counts: Vec<u64> = scene
        .meshes
        .iter()
        .map(|m| m.num_triangles as u64)
        .collect();

    // 【体积预算切分】b3dm 大小 ≈ 三角形数 × 63B（实测校准：ABC831 245 万三角形
    // = 152.6MB、未命名.fbx 同量级）。把目标 tile 大小折算成三角形预算做停止条件。
    //
    // 预算取"目标上限的 2 倍"：--tile-size-mb=10 时容忍到 ~20MB 才切——父格子
    // 12MB 就不切（省掉切分产生的一堆 <2MB 碎渣 tile），只有真超 20MB 的聚集簇
    // 才继续细分，切到每块 ≤~20MB。输出整体落在 2-20MB、文件数几十个。
    let max_tri_per_tile = (tile_size_mb * 1_000_000.0 / 63.0).max(1.0) as u64;
    let builder = OctreeBuilder {
        max_depth,
        max_meshes_per_leaf: max_meshes,
        // 只防极端"薄片豆腐"：水平 0.2m 以下不再切。不再承担"tile 大小"职责。
        min_extent: 0.5,
        max_triangles_per_tile: max_tri_per_tile,
    };
    let tree = builder.build(root_bounds, &mesh_centers, &mesh_tri_counts);
    eprintln!("[build] 八叉树构建完成，{} 个非空叶节点", tree.leaf_count());

    // 【cache-busting 版本戳】曾用构建时间戳生成 ?v= 参数，后改为"uri 不带查询串"
    // （查询串会让浏览器放弃磁盘缓存，tile 被卸载后回视口变成全量网络请求），
    // 改靠 Last-Modified/304 自然缓存。变量保留备查。
    let _build_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // 3. 收集要生成 tile 的节点
    //    lod == 0 → 只收叶节点、不简化（旧行为，单层）
    //    lod >= 1 → 收 LOD 金字塔的 top (lod+1) 层，层越粗简化越狠
    let mut specs: Vec<(&octree::OctreeNode, f32)> = Vec::new();
    // "小 tile 合并"pass 用：被合并的 leaf 在写 tileset.json 时跳过
    let mut dead: HashSet<usize> = HashSet::new();
    if lod == 0 {
        let mut leaves: Vec<&octree::OctreeNode> = Vec::new();
        tree.collect_leaves(&mut leaves);
        leaves.retain(|l| !l.mesh_ids.is_empty());
        for l in leaves {
            specs.push((l, 0.0));
        }
        eprintln!("[build] 单层模式：{} 个叶节点 tile（合并前）", specs.len());

        // ===== 小 tile 合并 pass =====
        // 八叉树按 mesh 中心点切分时会产生大量 <2MB 的稀疏 leaf（ABC831 48 tile 中有
        // 22 个 <2MB）。这些小 tile 在 Cesium frustum 边界会高频 evict → 拖动时不停
        // 重新 fetch。把 <2MB 的 leaf 自动合并到空间最近、合并后 ≤20MB 的大 leaf，
        // 目标 tile 数减到接近 mainview 26（实测从 48 → ~30 tile）。
        let enable_merge = true;
        let min_tri_per_tile = (2.0_f64 * 1_000_000.0 / 63.0) as u64;
        let max_tri = (tile_size_mb * 1_000_000.0 / 63.0).max(1.0) as u64;

        // 估算每个 leaf 的三角形数（含中间节点递归收集，但 leaf 直接拿自己 mesh_ids）
        let mut leaf_tris: HashMap<usize, u64> = HashMap::new();
        for &(leaf, _) in &specs {
            let key = leaf as *const octree::OctreeNode as usize;
            let tris: u64 = leaf.mesh_ids.iter().map(|&id| mesh_tri_counts[id]).sum();
            leaf_tris.insert(key, tris);
        }

        // 按三角形数从少到多排序（小 leaf 优先合并）
        let mut small_keys: Vec<usize> = leaf_tris
            .iter()
            .filter(|(_, &t)| t < min_tri_per_tile)
            .map(|(k, _)| *k)
            .collect();
        small_keys.sort_by_key(|k| leaf_tris[k]);

        for &small_key in &small_keys {
            if !enable_merge || dead.contains(&small_key) {
                continue;
            }
            let small_tris = leaf_tris[&small_key];
            let small_leaf: &octree::OctreeNode = unsafe { &*(small_key as *const octree::OctreeNode) };
            let sc = small_leaf.bounds.center();
            // 找空间最近、合并后 ≤ max_tri、且仍 alive 的大 leaf
            let mut best: Option<usize> = None;
            let mut best_dist = f64::INFINITY;
            for &(big_leaf, _) in &specs {
                let big_key = big_leaf as *const octree::OctreeNode as usize;
                if big_key == small_key || dead.contains(&big_key) {
                    continue;
                }
                let big_tris = leaf_tris[&big_key];
                if big_tris + small_tris > max_tri {
                    continue;
                }
                let bc = big_leaf.bounds.center();
                let dx = sc[0] - bc[0];
                let dy = sc[1] - bc[1];
                let dz = sc[2] - bc[2];
                let d = dx * dx + dy * dy + dz * dz;
                if d < best_dist {
                    best_dist = d;
                    best = Some(big_key);
                }
            }
            if let Some(big_key) = best {
                // 合并 mesh_ids 到 big leaf
                let big_leaf: &mut octree::OctreeNode =
                    unsafe { &mut *(big_key as *mut octree::OctreeNode) };
                big_leaf.mesh_ids.extend_from_slice(&small_leaf.mesh_ids);
                big_leaf.mesh_ids.sort_unstable();
                big_leaf.mesh_ids.dedup();
                // 更新 big 的 tris 估算
                leaf_tris.insert(big_key, leaf_tris[&big_key] + small_tris);
                dead.insert(small_key);
            }
        }

        let before_count = specs.len();
        specs.retain(|(l, _)| !dead.contains(&(*l as *const octree::OctreeNode as usize)));
        eprintln!(
            "[build] 小 tile 合并: {} → {} tile（合并 {} 个 <2MB tile）",
            before_count,
            specs.len(),
            dead.len()
        );
    } else {
        collect_lod_nodes(&tree, max_depth, lod, &mut specs);
        eprintln!(
            "[build] LOD 金字塔模式：{} 层，共 {} 个 tile",
            lod + 1,
            specs.len()
        );
    }

    // 4. 对每个节点：提取子树几何 → LOD 简化 → 写 b3dm
    let mut tile_count = 0;
    let mut total_size = 0u64;
    // 节点指针 -> (真实 AABB, 文件名)：用于生成 hierarchical tileset
    let mut node_bounds: HashMap<usize, ([f32; 3], [f32; 3])> = HashMap::new();
    let mut node_file: HashMap<usize, String> = HashMap::new();
    // tile 序号 i -> b3dm 字节数：单层 tileset 生成 cache-busting 版本参数用
    let mut tile_sizes: HashMap<usize, u64> = HashMap::new();

    for (i, (node, cell)) in specs.iter().enumerate() {
        // 该节点子树的全部 mesh（中间节点要收后代，才能生成覆盖整块的粗模型）
        let mut mesh_ids: Vec<usize> = Vec::new();
        if node.is_leaf() {
            mesh_ids.extend_from_slice(&node.mesh_ids);
        } else {
            node.collect_subtree_mesh_ids(&mut mesh_ids);
        }
        mesh_ids.sort_unstable();
        mesh_ids.dedup();

        // 按 per-face material 拆分：每个 sub-mesh 用自己真正的 baseColor 贴图
        let mut meshes = Vec::new();
        for &mesh_id in &mesh_ids {
            let m = &scene.meshes[mesh_id];
            let xform = mesh_to_world.get(&(m as *const _ as usize)).map(|mat| VertexXform {
                g2w: *mat,
                n2w: ufbx::matrix_for_normals(mat),
                scale: model_scale,
                off: model_off,
                flip_winding: linear_determinant(mat) < 0.0,
            });
            meshes.extend(extract_mesh_parts(m, input_dir, xform.as_ref(), metallic, roughness, mesh_id));
        }

        // LOD 简化（cell == 0 表示最细层，不动）
        if *cell > 0.0 {
            meshes = meshes.iter().map(|m| simplify_mesh(m, *cell)).collect();
        }

        // ===== pick 属性（Batch Table + _BATCHID）=====
        // 构件粒度 = 原 mesh（与 CesiumLab scenetree 的 element 一致）。
        // mesh_ids 已排序去重 → 位置即 tile 内局部 batchId。每个顶点记下自己构件的
        // 局部 id；构件名表（batch_names）供 Batch Table 生成 name/id 行。
        // 注意在 merge 之前填：merge 按顶点段拼接 batch_ids，合并后 primitive 内
        // 各构件的 batchId 仍然互不相同，pick 才能区分到构件。
        let mesh_gid_to_local: HashMap<usize, u32> = mesh_ids.iter().enumerate()
            .map(|(local, &gid)| (gid, local as u32)).collect();
        let batch_names: Vec<String> = mesh_ids.iter().map(|&gid| {
            let m = &scene.meshes[gid];
            let mut n = m.element.name.to_string();
            if n.is_empty() {
                n = mesh_node_names.get(&(m as *const ufbx::Mesh as usize)).cloned().unwrap_or_default();
            }
            if n.is_empty() { format!("mesh_{}", gid) } else { n }
        }).collect();
        let batch_gids: Vec<usize> = mesh_ids.clone();
        for part in meshes.iter_mut() {
            let local = mesh_gid_to_local.get(&part.feature_gid).copied().unwrap_or(0);
            part.batch_ids = vec![local; part.positions.len()];
        }

        // 同贴图的 mesh 合并成一个 primitive：4111 → ~材质数，draw call 骤降（帧率关键）
        let meshes = merge_meshes_by_texture(meshes);

        let geometry = TileGeometry { meshes, batch_names, batch_gids };
        if geometry.total_vertices() == 0 || geometry.total_triangles() == 0 {
            eprintln!("[build] 跳过 tile {} (空几何)", i);
            continue;
        }

        // 真实几何 AABB（用顶点算，八叉树格子包不住大 mesh 的外伸顶点）
        let gb = match geometry.compute_aabb_all() {
            Some(v) => v,
            None => {
                eprintln!("[build] 跳过 tile {} (拿不到 AABB)", i);
                continue;
            }
        };

        // geometricError：叶节点 0，越粗的层误差越大（Cesium 据此决定何时细化）
        let ge = if node.is_leaf() {
            0.0
        } else {
            node.bounds.max_extent() / 8.0
        };

        let b3dm_bytes = build_b3dm(&geometry, ge)?;
        // 文件名：单层模式保持 NoLod_<i>.b3dm（兼容旧输出），LOD 模式用 L<depth>_<i>.b3dm
        let fname = if lod == 0 {
            format!("NoLod_{}.b3dm", i)
        } else {
            format!("L{}_{}.b3dm", node.depth, i)
        };
        std::fs::write(output.join(&fname), &b3dm_bytes)?;

        let key = *node as *const octree::OctreeNode as usize;
        node_bounds.insert(key, gb);
        // uri 不带 ?v= 版本参数：查询串会让浏览器放弃磁盘缓存，
        // tile 被卸载后回视口就变成全量网络请求。改靠 Last-Modified/304 自然缓存。
        node_file.insert(key, fname.clone());
        tile_sizes.insert(i, b3dm_bytes.len() as u64);

        tile_count += 1;
        total_size += b3dm_bytes.len() as u64;

        if tile_count <= 8 || tile_count % 20 == 0 {
            eprintln!(
                "[build]  [{}] depth={} mesh={} 三角形={} cell={:.2} b3dm={:.2} KB ge={:.3}",
                i,
                node.depth,
                geometry.meshes.len(),
                geometry.total_triangles(),
                cell,
                b3dm_bytes.len() as f64 / 1024.0,
                ge,
            );
        }
    }
    // 每 tile 大小分布汇总（验证 2-20MB 目标区间）
    if !tile_sizes.is_empty() {
        let mut sizes: Vec<f64> = tile_sizes.values().map(|&b| b as f64 / 1048576.0).collect();
        sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sizes.len();
        let min_mb = sizes[0];
        let max_mb = sizes[n - 1];
        let avg_mb = sizes.iter().sum::<f64>() / n as f64;
        let in_range = sizes.iter().filter(|&&m| (2.0..=20.0).contains(&m)).count();
        let over = sizes.iter().filter(|&&m| m > 20.0).count();
        let under = sizes.iter().filter(|&&m| m < 2.0).count();
        eprintln!(
            "[build] tile 大小: min={:.2}MB max={:.2}MB avg={:.2}MB | 2-20MB 内 {}/{}（<2MB:{}，>20MB:{}）",
            min_mb, max_mb, avg_mb, in_range, n, under, over
        );
    }
    eprintln!(
        "[build] tile 写完成: {} 个，总大小 {:.2} MB",
        tile_count,
        total_size as f64 / 1024.0 / 1024.0
    );

    // 5. 写 tileset.json
    let tileset = if lod == 0 {
        // 单层：root 直接挂所有 tile
        // 跳过被"小 tile 合并"pass 标记为 dead 的 leaf（它们的 mesh_ids 已并到其他 leaf）
        let mut leaves = Vec::new();
        tree.collect_leaves(&mut leaves);
        leaves.retain(|l| !l.mesh_ids.is_empty() && !dead.contains(&(*l as *const octree::OctreeNode as usize)));
        let mut ordered: Vec<Option<([f32; 3], [f32; 3])>> = Vec::with_capacity(leaves.len());
        for l in &leaves {
            ordered.push(node_bounds.get(&(*l as *const octree::OctreeNode as usize)).copied());
        }
        build_tileset(&tree, tile_count, &ordered, &tile_sizes, &dead)
    } else {
        build_tileset_hierarchical(&tree, max_depth, lod, &node_bounds, &node_file)
    };
    let tileset_path = output.join("tileset.json");

    // 把模型定位参数写入 tileset.json 的 asset.location，
    // demo.html 加载 tileset 后读取这里，自动生成 modelMatrix。
    // 不传经纬度时使用默认北京天安门（116.397428, 39.90923）。
    let mut tileset = tileset;
    let location = serde_json::json!({
        "longitude": longitude,
        "latitude": latitude,
        "heading": heading,
        "pitch": pitch,
        "roll": roll,
    });

    // 必须确保 "asset" 在顶层 Map 里以 object 形式存在，再注入 location。
    // 用 Entry API 避免 serde_json 的 Index panic（键不存在会崩溃）。
    let asset_obj = match tileset.get_mut("asset") {
        Some(a) => match a.as_object_mut() {
            Some(m) => m,
            None => {
                *a = serde_json::json!({});
                a.as_object_mut().expect("just initialized")
            }
        },
        None => {
            tileset
                .as_object_mut()
                .expect("tileset must be an object")
                .insert("asset".to_string(), serde_json::json!({}));
            tileset
                .get_mut("asset")
                .expect("just inserted")
                .as_object_mut()
                .expect("must be object")
        }
    };
    asset_obj.insert("location".to_string(), location);

    std::fs::write(&tileset_path, serde_json::to_string_pretty(&tileset)?)?;
    eprintln!("[build] tileset.json 已写入: {}", tileset_path.display());

    // 6. 写 scenetree.json（对齐 CesiumLab 格式：构件树 + ECEF 定位球，pick 后互查用）
    write_scenetree(
        &scene, &mesh_to_world, &mesh_node_names, model_scale, model_off,
        longitude, latitude, heading, &output,
    )?;

    Ok(())
}

/// 计算 WGS84 椭球上某经纬度的 ECEF 坐标
fn wgs84_ecef(lon_deg: f64, lat_deg: f64, h: f64) -> [f64; 3] {
    let a = 6378137.0_f64;
    let e2 = 6.69437999014e-3_f64;
    let lon = lon_deg.to_radians();
    let lat = lat_deg.to_radians();
    let n = a / (1.0 - e2 * lat.sin() * lat.sin()).sqrt();
    [
        (n + h) * lat.cos() * lon.cos(),
        (n + h) * lat.cos() * lon.sin(),
        (n * (1.0 - e2) + h) * lat.sin(),
    ]
}

/// ENU 东北天偏移量 → ECEF 偏移向量
fn enu_offset_to_ecef(lon_deg: f64, lat_deg: f64, e: f64, n: f64, u: f64) -> [f64; 3] {
    let lon = lon_deg.to_radians();
    let lat = lat_deg.to_radians();
    let (sl, cl) = lon.sin_cos();
    let (sp, cp) = lat.sin_cos();
    // east 单位向量 (-sinλ, cosλ, 0)、north (-sinφcosλ, -sinφsinλ, cosφ)、up (cosφcosλ, cosφsinλ, sinφ)
    [
        e * -sl + n * -sp * cl + u * cp * cl,
        e * cl + n * -sp * sl + u * cp * sl,
        n * cp + u * sp,
    ]
}

/// 生成 scenetree.json：CesiumLab mainview 同构（scenes[0] root + element 平铺）
///
/// - children = 每个原 mesh 一条 {id, name, sphere, type:"element"}
/// - sphere = [ECEF x, y, z, r]：构件 AABB 中心经 heading 旋转 + ENU→ECEF 后的球
/// - id 与 b3dm Batch Table 的 id 同算法（FNV-1a(name#g全局序号)），pick 命中后
///   可直接在 scenetree 里定位构件（树控件高亮/定位用）
fn write_scenetree(
    scene: &ufbx::Scene,
    mesh_to_world: &HashMap<usize, ufbx::Matrix>,
    mesh_node_names: &HashMap<usize, String>,
    model_scale: f64,
    model_off: [f64; 3],
    longitude: f64,
    latitude: f64,
    heading: f64,
    output: &std::path::Path,
) -> Result<()> {
    let h = heading.to_radians();
    // f64::sin_cos 返回 (sin, cos)——别把顺序当反（反了旋转错 92°，sphere 全偏）
    let (sin_h, cos_h) = h.sin_cos();
    let anchor = wgs84_ecef(longitude, latitude, 0.0);

    let mut children: Vec<serde_json::Value> = Vec::with_capacity(scene.meshes.len());
    // 场景整体范围（root sphere 用）
    let mut root_min = [f64::INFINITY; 3];
    let mut root_max = [f64::NEG_INFINITY; 3];

    for (gid, m) in scene.meshes.iter().enumerate() {
        // 构件名 fallback 链：mesh.element.name → node 名 → mesh_<gid>（与 Batch Table 一致）
        let mm: &ufbx::Mesh = m; // Ref<Mesh> → &Mesh（指针键与 build_cmd 的映射一致）
        let name = {
            let mut n = m.element.name.to_string();
            if n.is_empty() {
                n = mesh_node_names.get(&(mm as *const ufbx::Mesh as usize)).cloned().unwrap_or_default();
            }
            if n.is_empty() { format!("mesh_{}", gid) } else { n }
        };

        // 构件世界局部 AABB：全部顶点经 g2w（node 变换）→ 缩放 → 贴地/原点偏移
        // （与切片的 xform 变换链一致：transform_position(g2w) * scale - off，Z-up 输出系）
        let mat = mesh_to_world.get(&(mm as *const ufbx::Mesh as usize));
        let (mut mn, mut mx) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        let vp = &m.vertex_position;
        for i in 0..vp.values.len() {
            let p = unsafe { vp.values.data.add(i).read() };
            // 与切片顶点完全一致：transform_position(g2w) * scale - off，直出不翻轴
            let pw = match mat {
                Some(x) => ufbx::transform_position(x, p),
                None => p,
            };
            let out = [
                pw.x * model_scale - model_off[0],
                pw.y * model_scale - model_off[1],
                pw.z * model_scale - model_off[2],
            ];
            for k in 0..3 {
                mn[k] = mn[k].min(out[k]);
                mx[k] = mx[k].max(out[k]);
            }
        }
        if !mn[0].is_finite() {
            continue; // 空网格跳过
        }
        let center = [(mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, (mn[2] + mx[2]) / 2.0];
        // 包围球半径 = 对角线一半
        let radius = ((mx[0] - mn[0]).powi(2) + (mx[1] - mn[1]).powi(2) + (mx[2] - mn[2]).powi(2)).sqrt() / 2.0;
        for k in 0..3 {
            root_min[k] = root_min[k].min(mn[k]);
            root_max[k] = root_max[k].max(mx[k]);
        }

        // 局部 Z-up → ENU（Cesium heading：从北顺时针）
        let east = center[0] * cos_h + center[1] * sin_h;
        let north = -center[0] * sin_h + center[1] * cos_h;
        let up = center[2];
        let off = enu_offset_to_ecef(longitude, latitude, east, north, up);
        let ecef = [
            anchor[0] + off[0],
            anchor[1] + off[1],
            anchor[2] + off[2],
        ];

        // id 与 Batch Table 同算法
        let id_src = format!("{}#g{}", name, gid);
        let mut hv: u32 = 0x811c9dc5;
        for b in id_src.bytes() {
            hv ^= b as u32;
            hv = hv.wrapping_mul(0x01000193);
        }
        let id = format!("{:08x}{:08x}{:08x}{:08x}", hv, hv.wrapping_add(gid as u32), 0u32, 0u32);

        children.push(serde_json::json!({
            "id": id,
            "name": name,
            "sphere": [ecef[0], ecef[1], ecef[2], radius.max(0.01)],
            "type": "element"
        }));
    }

    // root sphere
    let rc = [
        (root_min[0] + root_max[0]) / 2.0,
        (root_min[1] + root_max[1]) / 2.0,
        (root_min[2] + root_max[2]) / 2.0,
    ];
    let rr = ((root_max[0] - root_min[0]).powi(2)
        + (root_max[1] - root_min[1]).powi(2)
        + (root_max[2] - root_min[2]).powi(2)).sqrt() / 2.0;
    let reast = rc[0] * cos_h + rc[1] * sin_h;
    let rnorth = -rc[0] * sin_h + rc[1] * cos_h;
    let roff = enu_offset_to_ecef(longitude, latitude, reast, rnorth, rc[2]);
    let root_id_src = format!("root#g0");
    let mut hv: u32 = 0x811c9dc5;
    for b in root_id_src.bytes() {
        hv ^= b as u32;
        hv = hv.wrapping_mul(0x01000193);
    }

    let tree = serde_json::json!({
        "scenes": [{
            "id": format!("{:08x}{:08x}{:08x}{:08x}", hv, hv.wrapping_add(0u32), 0u32, 0u32),
            "name": "root",
            "type": "node",
            "sphere": [anchor[0] + roff[0], anchor[1] + roff[1], anchor[2] + roff[2], rr.max(0.01)],
            "children": children
        }]
    });
    let path = output.join("scenetree.json");
    std::fs::write(&path, serde_json::to_string(&tree)?)?;
    eprintln!("[build] scenetree.json 已写入: {}（构件 {} 个）", path.display(), children.len());
    Ok(())
}

/// LOD 层号 → 顶点聚类网格边长（米）
///
/// 层号 0 = 最细（八叉树叶节点，不简化）；数字越大层越粗。
/// 每粗一级网格放大约 4 倍，三角形数通常降到上一级的 1/3 ~ 1/5。
fn lod_cell_size(lod_layer: u32) -> f32 {
    match lod_layer {
        0 => 0.0,   // 最细层：不简化，保留全部几何细节
        1 => 0.35,
        2 => 1.5,
        3 => 6.0,
        _ => 24.0,
    }
}

/// 返回 node 子树中最深叶节点的 depth
///
/// 八叉树并不总是细分到 max_depth：某块 mesh 数 <= max_meshes 时会提前停止。
/// 所以判断"这个节点离最细层还有多远"必须看子树实际深度，不能用 max_depth 硬算。
fn subtree_leaf_depth(node: &octree::OctreeNode) -> u32 {
    if node.is_leaf() {
        return node.depth;
    }
    let mut deepest = node.depth;
    for c in &node.children {
        deepest = deepest.max(subtree_leaf_depth(c));
    }
    deepest
}

/// 递归收集 LOD 金字塔中要生成 tile 的节点
///
/// 只收子树非空的节点，且层号（= max_depth - depth）不超过 lod_levels。
fn collect_lod_nodes<'a>(
    node: &'a octree::OctreeNode,
    max_depth: u32,
    lod_levels: u32,
    out: &mut Vec<(&'a octree::OctreeNode, f32)>,
) {
    let mut ids = Vec::new();
    node.collect_subtree_mesh_ids(&mut ids);
    if ids.is_empty() {
        return;
    }
    // 所有非空节点都必须有 content —— 否则浅层的**叶节点**（八叉树因 mesh 数少
    // 而提前停止细分，本身直接持有 mesh）会因为层号超出 LOD 层数而没有 content，
    // 整块几何直接从结果里消失（实测丢了 13 个叶节点、30 张贴图）。
    //
    // 层号 = 本节点到子树最深叶节点的高度，**不能用 max_depth - depth**：
    // 八叉树会提前停止细分，depth=3 就可能已经是叶节点。若按 max_depth 硬算，
    // 这些浅层叶节点会被当成粗层用 cell=0.35 简化，最精细的一层几何精度直接
    // 被砍掉（实测最细层只剩 84 万三角形，而全精度应有 253 万）。
    // 叶节点高度 h=0 -> cell=0，即不简化。
    let deepest = subtree_leaf_depth(node);
    let h = deepest.saturating_sub(node.depth);
    let _ = max_depth;
    let cell = lod_cell_size(h.min(lod_levels));
    out.push((node, cell));
    for c in &node.children {
        collect_lod_nodes(c, max_depth, lod_levels, out);
    }
}

/// 生成 hierarchical tileset.json（真正的 LOD 金字塔）
///
/// 结构：root → children 递归，每个节点带自己的 content（该块的简化模型）
/// 和指向更细层 children 的引用。geometricError 逐层递减，
/// Cesium 据此在相机靠近时自动把粗模型替换成细模型。
fn build_tileset_hierarchical(
    root: &octree::OctreeNode,
    max_depth: u32,
    lod_levels: u32,
    node_bounds: &HashMap<usize, ([f32; 3], [f32; 3])>,
    node_file: &HashMap<usize, String>,
) -> serde_json::Value {
    fn rec(
        node: &octree::OctreeNode,
        max_depth: u32,
        lod_levels: u32,
        node_bounds: &HashMap<usize, ([f32; 3], [f32; 3])>,
        node_file: &HashMap<usize, String>,
        rmin: &mut [f32; 3],
        rmax: &mut [f32; 3],
    ) -> Option<(serde_json::Value, [f32; 3], [f32; 3])> {
        let key = node as *const octree::OctreeNode as usize;

        // 递归处理子节点
        let mut children = Vec::new();
        // 所有子节点的 AABB 并集，用来保证父 box 一定包得住子 box
        let mut cmin = [f32::INFINITY; 3];
        let mut cmax = [f32::NEG_INFINITY; 3];
        for c in &node.children {
            if let Some((cj, kmin, kmax)) =
                rec(c, max_depth, lod_levels, node_bounds, node_file, rmin, rmax)
            {
                for k in 0..3 {
                    if kmin[k] < cmin[k] {
                        cmin[k] = kmin[k];
                    }
                    if kmax[k] > cmax[k] {
                        cmax[k] = kmax[k];
                    }
                }
                children.push(cj);
            }
        }

        // 本节点的包围盒 = 自身真实几何 AABB ∪ 所有子节点的 AABB。
        //
        // **必须取并集**：父节点存的是简化网格，简化会丢掉外围的细小三角形，
        // 父几何 AABB 可能反而比子几何 AABB 小。一旦父 box 包不住子 box，
        // Cesium 视锥剔除会先判父 box 不可见就把整棵子树剔掉 —— 表现就是
        // "相机靠近时表面模型成片消失，拉远又回来"。
        let mut bmin = [f32::INFINITY; 3];
        let mut bmax = [f32::NEG_INFINITY; 3];
        let mut have = false;
        if let Some(g) = node_bounds.get(&key).copied() {
            bmin = g.0;
            bmax = g.1;
            have = true;
        }
        if !children.is_empty() {
            if !have {
                bmin = cmin;
                bmax = cmax;
                have = true;
            } else {
                for k in 0..3 {
                    if cmin[k] < bmin[k] {
                        bmin[k] = cmin[k];
                    }
                    if cmax[k] > bmax[k] {
                        bmax[k] = cmax[k];
                    }
                }
            }
        }
        if !have {
            return None;
        }
        let bounds = (bmin, bmax);

        for k in 0..3 {
            if bounds.0[k] < rmin[k] {
                rmin[k] = bounds.0[k];
            }
            if bounds.1[k] > rmax[k] {
                rmax[k] = bounds.1[k];
            }
        }

        // 该节点是否有 content：只要构建阶段为它写过 b3dm 就有
        // （所有非空节点都会写，浅层节点用最粗简化级别）
        let _ = (max_depth, lod_levels);
        let has_content = node_file.contains_key(&key);

        // 没有任何子节点也没有 content → 这个节点没意义
        if children.is_empty() && !has_content {
            return None;
        }

        // geometricError：叶节点 0，否则按块尺寸给（越粗越大）
        let ge = if node.is_leaf() {
            0.0
        } else {
            node.bounds.max_extent() / 8.0
        };

        let mut obj = serde_json::Map::new();
        obj.insert(
            "boundingVolume".to_string(),
            serde_json::json!({ "box": aabb_minmax_to_box12_padded(bounds.0, bounds.1) }),
        );
        obj.insert("geometricError".to_string(), serde_json::json!(ge));
        obj.insert("refine".to_string(), serde_json::json!("REPLACE"));
        if has_content {
            obj.insert(
                "content".to_string(),
                serde_json::json!({ "uri": node_file[&key] }),
            );
        }
        if !children.is_empty() {
            obj.insert("children".to_string(), serde_json::Value::Array(children));
        }
        Some((serde_json::Value::Object(obj), bmin, bmax))
    }

    let mut rmin = [f32::INFINITY; 3];
    let mut rmax = [f32::NEG_INFINITY; 3];
    let (root_json, _rbmin, _rbmax) = rec(
        root,
        max_depth,
        lod_levels,
        node_bounds,
        node_file,
        &mut rmin,
        &mut rmax,
    )
    .expect("root 节点必须有内容");

    let root_extent = ((rmax[0] - rmin[0])
        .max(rmax[1] - rmin[1])
        .max(rmax[2] - rmin[2])) as f64;

    serde_json::json!({
        "asset": {
            "generatetool": "RustFBX2Tiles",
            "version": "1.0"
        },
        "geometricError": root_extent,
        "refine": "REPLACE",
        "root": root_json
    })
}

/// 顶点输出变换：p_out = transform_position(g2w, p) * scale - off
/// 法线输出：transform_direction(n2w, n) 后归一化
struct VertexXform {
    g2w: ufbx::Matrix,
    n2w: ufbx::Matrix,
    scale: f64,
    off: [f64; 3],
    flip_winding: bool,
}

fn linear_determinant(m: &ufbx::Matrix) -> f64 {
    m.m00 * (m.m11 * m.m22 - m.m12 * m.m21)
        - m.m01 * (m.m10 * m.m22 - m.m12 * m.m20)
        + m.m02 * (m.m10 * m.m21 - m.m11 * m.m20)
}

/// 贴图压缩管线：解码 → 超过 256 降采样 → 按是否真有 alpha 编码 PNG/JPEG(q80)。
///
/// 【为什么必须压缩】实测同一模型：CesiumLab 贴图总量 15.1MB（190 张，JPEG 平均
/// 47KB），直接拷贝原始贴图则 286.5MB——GPU 显存被撑爆，帧率从 100+ 掉到 20-30，
/// 模型离开视口再回来时重新解码几十上百 MB 卡死。压缩后显存与 CesiumLab 同量级。
///
/// 【为什么压到 256 而不是 512/1024】贴图 GPU 占用 = width × height × 4 字节（RGBA）。
/// 1024 = 4MB/张，240 张 = 960MB GPU 资源——超浏览器/GPU 限制触发 tile evict。
/// 压到 256：每张 256KB，总 60MB GPU 贴图——加上 ~60MB 顶点 buffer，
/// 总 GPU 占用 ~120MB 远低于 Cesium 内部 cache 上限，拖动可 0 重 fetch。
/// 视觉损失：油库远景几乎看不出，近景屋顶纹理用 mipmap 链补足。
fn process_texture(bytes: &[u8]) -> Option<(Vec<u8>, String, bool)> {
    let img = image::load_from_memory(bytes).ok()?;
    let (w, h) = (img.width(), img.height());
    let img = if w.max(h) > 256 {
        let s = 256.0_f32 / w.max(h) as f32;
        let nw = ((w as f32 * s).round() as u32).max(1);
        let nh = ((h as f32 * s).round() as u32).max(1);
        img.resize(nw, nh, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    // 真 alpha 检测：存在 alpha<250 的像素才算透明（RGBA 但全不透明的图转 JPEG 省一半以上体积）
    let rgba = img.to_rgba8();
    let mut has_alpha = false;
    for px in rgba.pixels() {
        if px.0[3] < 250 {
            has_alpha = true;
            break;
        }
    }
    if has_alpha {
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(rgba)
            .write_to(&mut out, image::ImageFormat::Png)
            .ok()?;
        Some((out.into_inner(), "image/png".to_string(), true))
    } else {
        let rgb = img.to_rgb8();
        let mut out = std::io::Cursor::new(Vec::new());
        let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 80);
        image::DynamicImage::ImageRgb8(rgb)
            .write_with_encoder(enc)
            .ok()?;
        Some((out.into_inner(), "image/jpeg".to_string(), false))
    }
}

/// 同一贴图可能被上百个 mesh 引用，压缩结果全局缓存，避免重复解码/缩放。
fn texture_cache() -> &'static std::sync::Mutex<HashMap<String, std::sync::Arc<(Vec<u8>, String, bool)>>> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<String, std::sync::Arc<(Vec<u8>, String, bool)>>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// 贴图源文件查找：先按 FBX 记录的路径（相对 input_dir 或绝对），找不到就按
/// 文件名 basename 在 input_dir、input_dir 父目录及其 *.fbm 子目录里回退搜索。
///
/// 【为什么要回退】FBX 常记录烘焙时的绝对路径（如 `C:/xxx/Desktop/AB/贴图.jpg`），
/// 换机器/挪目录后路径失效。同名贴图往往就在模型旁的 .fbm 目录里，按 basename
/// 找回即可正常贴图（实测 ABC831.fbx 引用 Desktop/AB 而贴图实际在 Desktop/未命名.fbm）。
fn resolve_texture_path(tex_filename: &str, input_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let direct = if std::path::Path::new(tex_filename).is_absolute() {
        std::path::PathBuf::from(tex_filename)
    } else {
        input_dir.join(tex_filename)
    };
    if direct.exists() {
        return Some(direct);
    }
    // 回退：按 basename 在候选目录里找（含大小写不敏感兜底）
    let base = std::path::Path::new(tex_filename).file_name()?.to_string_lossy().to_string();
    let base_lower = base.to_lowercase();
    // 候选目录：input_dir、input_dir 父目录、这两者下的 *.fbm 子目录
    let mut dirs: Vec<std::path::PathBuf> = vec![input_dir.to_path_buf()];
    if let Some(p) = input_dir.parent() {
        dirs.push(p.to_path_buf());
        if let Some(pp) = p.parent() {
            dirs.push(pp.to_path_buf());
        }
    }
    let mut fbm_dirs: Vec<std::path::PathBuf> = Vec::new();
    for d in &dirs {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() && p.extension().map(|x| x.to_string_lossy().to_lowercase() == "fbm").unwrap_or(false) {
                    fbm_dirs.push(p);
                }
            }
        }
    }
    for d in dirs.iter().chain(fbm_dirs.iter()) {
        let cand = d.join(&base);
        if cand.exists() {
            return Some(cand);
        }
        // 大小写兜底（.JPG vs .jpg）
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                if let Ok(name) = e.file_name().into_string() {
                    if name.to_lowercase() == base_lower {
                        return Some(e.path());
                    }
                }
            }
        }
    }
    None
}

/// 从单个 material 抽取 baseColor 贴图（压缩后），
/// 返回 `(贴图名, 是否含 alpha, 压缩后字节, mime)`，没贴图返回 None
///
/// 贴图源路径 = FBX 内嵌的 tex.filename（通常是 .fbm 子目录里的相对路径）。
fn extract_material_texture(
    material: &ufbx::Material,
    input_dir: &std::path::Path,
) -> Option<(String, bool, Vec<u8>, String)> {
    let tex = material.pbr.base_color.texture.as_ref()?;
    if tex.filename.is_empty() {
        return None;
    }
    let tex_filename = tex.filename.to_string();
    let src_path = resolve_texture_path(&tex_filename, input_dir);
    let src_path = match src_path {
        Some(p) => p,
        None => {
            eprintln!("[tex] 找不到贴图源(已回退搜索): {}", tex_filename);
            return None;
        }
    };
    let cache_key = src_path.to_string_lossy().to_string();
    let entry = {
        let mut cache = texture_cache().lock().unwrap();
        cache.entry(cache_key).or_insert_with(|| {
            let raw = std::fs::read(&src_path).unwrap_or_default();
            std::sync::Arc::new(process_texture(&raw).unwrap_or_else(|| {
                (raw.clone(), "application/octet-stream".to_string(), false)
            }))
        }).clone()
    };
    let (bytes, mime, has_alpha) = (&entry.0, &entry.1, &entry.2);
    if *has_alpha {
        eprintln!("[tex] 含真 alpha（alphaMode=MASK，保留 PNG）: {}", src_path.display());
    }
    let name = src_path.file_name()?.to_string_lossy().to_string();
    Some((name, *has_alpha, bytes.clone(), mime.clone()))
}

/// 从 ufbx::Mesh 提取几何数据 —— **按 per-face material 拆分成多个 sub-mesh**
///
/// 【贴图错误根因修复】FBX 里一个 mesh 的 face 常横跨多个 material：
/// 本模型 2217 个 mesh 中有 1215 个（54.8%）是多材质的。之前只取 `materials.first()`
/// 会把整个 mesh 的所有面刷成第一张贴图 —— 这就是"所有面都变浅蓝灰"的根因。
///
/// 现在按 `mesh.face_material`（每 face 一个 material index）分组，
/// 每组生成一个独立 MeshData（→ glTF 里独立 primitive + 独立 material），
/// 各自用自己的 baseColor 贴图。
///
/// 顶点按 face 顺序逐点提取（不共享 vertex），并做 fan triangulation（n-gon → n-2 三角）。
/// ufbx 0.11 的 stream values.data 是 `*const T` 裸指针（FFI 绑定），需 unsafe 读。
fn extract_mesh_parts(
    m: &ufbx::Mesh,
    input_dir: &std::path::Path,
    xform: Option<&VertexXform>,
    metallic_override: f64,
    roughness_override: f64,
    feature_gid: usize,
) -> Vec<MeshData> {
    // 1. 按 face 的 material index 分组：mat_index -> [face_index...]
    //    BTreeMap 保证输出顺序稳定（同一次构建结果可复现）
    let mut groups: std::collections::BTreeMap<u32, Vec<usize>> =
        std::collections::BTreeMap::new();
    if m.face_material.len() == m.faces.len() && !m.face_material.is_empty() {
        for (fi, &mi) in m.face_material.iter().enumerate() {
            groups.entry(mi).or_default().push(fi);
        }
    } else {
        // 没有 per-face 材质信息（或长度对不上）：整个 mesh 作为一组
        groups.insert(0, (0..m.faces.len()).collect());
    }

    // 2. 每组 → 一个 MeshData
    let mut out: Vec<MeshData> = Vec::with_capacity(groups.len());
    for (mat_idx, face_list) in &groups {
        if face_list.is_empty() {
            continue;
        }

        // material：mat_idx 越界（face 未绑材质）时回退到第一个 material
        let material: Option<&ufbx::Material> = if (*mat_idx as usize) < m.materials.len() {
            Some(&m.materials[*mat_idx as usize])
        } else {
            m.materials.first().map(|r| r.as_ref())
        };

        let (texture_uri, texture_has_alpha, texture_bytes, texture_mime) = material
            .and_then(|mat| extract_material_texture(mat, input_dir))
            .map(|(name, a, bytes, mime)| (Some(name), a, Some(bytes), Some(mime)))
            .unwrap_or((None, false, None, None));

// 【反射效果】材质 metallic/roughness 用 CLI 传入值覆盖 FBX 原值。
    //
    //   为什么覆盖 FBX 原值：实测 ABC831.fbx 自带 roughness=0.859（接近全漫反射）
    //   → 模型表面没有环境高光/反射，观感发闷；而 CesiumLab mainview 输出固定
    //   roughness=0.45 → 表面有明显环境反射。所以默认用 CLI 值（0.45）覆盖 FBX，
    //   想忠实还原 FBX 材质时传 `--roughness 0.859`。
    //   （FBX 的 pbr.metalness/roughness 读取逻辑已删：CLI 值始终覆盖，读它无用。
    //     ufbx 兜底默认：roughness=1.0（全漫反射）、metallic=0.0——和 glTF 一致。）
    let metallic = metallic_override;
    let roughness = roughness_override;

        let name = format!("{}_m{}", m.element.name, mat_idx);
        // 透明通道只决定材质 alphaMode，不参与几何方向判断。
        // 几何是否包含镜像，只由 geometry_to_world 的线性部分行列式决定。
        let flip_winding = xform.map(|x| x.flip_winding).unwrap_or(false);

        if let Some(part) = extract_faces(m, face_list, texture_uri, texture_bytes, texture_mime, texture_has_alpha, xform, flip_winding, name) {
            let mut part = part;
            part.metallic_factor = metallic as f32;
            part.roughness_factor = roughness as f32;
            part.feature_gid = feature_gid;
            out.push(part);
        }
    }
    out
}

/// 顶点去重用的哈希 key（量化后的 position + normal + uv）
///
/// 【为什么要去重】P1 阶段为了让 UV/NORMAL 正确，改成了"每个 face 顶点独立存"，
/// 相邻 face 的公共顶点被重复存了 3~6 次，导致 b3dm 体积膨胀 3.4×（76 MB → 260 MB）。
/// 260 MB 一次性灌给 Cesium，超出其 tile 缓存预算后会**卸载当前视角外的 tile** ——
/// 表现就是"放大时局部模型消失，缩小后又出现"。
///
/// 按 (pos, nrm, uv) 三元组量化成 i32 做 key，完全相同的顶点合并成一个。
#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct VKey([i32; 8]);

impl VKey {
    fn new(pos: [f32; 3], nrm: [f32; 3], uv: [f32; 2]) -> Self {
        // 坐标量化精度 1e-4 m（0.1 mm），足以区分不同顶点且避免浮点误差导致漏合并
        let q = |v: f32| -> i32 { (v * 1.0e4).round() as i32 };
        VKey([
            q(pos[0]), q(pos[1]), q(pos[2]),
            q(nrm[0]), q(nrm[1]), q(nrm[2]),
            q(uv[0]),  q(uv[1]),
        ])
    }
}

// 不再使用尺寸推断高度轴：该 FBX 的树节点轴向已由 geometry_to_world 矩阵确认。

/// 提取指定 face 集合的几何（POSITION/NORMAL/UV + fan triangulation + 顶点去重）
///
/// 供 `extract_mesh_parts` 按材质分组后逐组调用。
fn extract_faces(
    m: &ufbx::Mesh,
    face_list: &[usize],
    texture_uri: Option<String>,
    texture_bytes: Option<Vec<u8>>,
    texture_mime: Option<String>,
    texture_has_alpha: bool,
    xform: Option<&VertexXform>,
    flip_winding: bool,
    name: String,
) -> Option<MeshData> {
    let has_normal = !m.vertex_normal.values.data.is_null();
    let has_uv = !m.vertex_uv.values.data.is_null();

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    // (量化后的顶点元组) -> 已分配的顶点下标
    let mut dedup: HashMap<VKey, u32> = HashMap::new();

    for &face_index in face_list {
        let face = &m.faces[face_index];
        let begin = face.index_begin as usize;
        let end = begin + face.num_indices as usize;
        let n_verts = end - begin;

        if n_verts < 3 {
            continue; // 退化的 face（点/线）跳过
        }

        // 该 face 去重后的顶点下标
        let mut face_verts: Vec<u32> = Vec::with_capacity(n_verts);

        for i in begin..end {
            // POSITION —— geometry 空间 → 场景坐标（g2w）→ 米（×0.01）→ 贴地（Y-up 输出）
            //
            // 【顶点必须 Y-up】Cesium 1.95 渲染 b3dm 的 glTF 时固定做 Y-up→Z-up 转换
            // （实验证实：Z-up 顶点会被转躺）。所以顶点保持 Y-up，让 Cesium 转换后立着。
            // boundingVolume 的对齐在 aabb_minmax_to_box12 里单独做 Y-up→Z-up 预转换。
            let p_idx = m.vertex_position.indices[i] as usize;
            let p = unsafe { m.vertex_position.values.data.add(p_idx).read() };
            let pos = match xform {
                Some(x) => {
                    let v = ufbx::transform_position(&x.g2w, p);
                    // 临时回退到 162MB 版本（不加 Z→Y 翻转）：(v.x, v.y, v.z) 直接出。
                    // 162MB 是用户确认"能显示模型"的版本——这是回退基准。
                    [(v.x * x.scale - x.off[0]) as f32,
                     (v.y * x.scale - x.off[1]) as f32,
                     (v.z * x.scale - x.off[2]) as f32]
                }
                None => [p.x as f32, p.y as f32, p.z as f32],
            };

            // NORMAL —— 法线用 for-normals 矩阵（不含平移），变换后归一化（不加 Z→Y 翻转）
            let nrm = if has_normal {
                let n_idx = m.vertex_normal.indices[i] as usize;
                let n = unsafe { m.vertex_normal.values.data.add(n_idx).read() };
                match xform {
                    Some(x) => {
                        let d = ufbx::transform_direction(&x.n2w, n);
                        let len = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
                        if len > 1e-12 {
                            [(d.x / len) as f32, (d.y / len) as f32, (d.z / len) as f32]
                        } else {
                            [0.0_f32, 1.0, 0.0]
                        }
                    }
                    None => [n.x as f32, n.y as f32, n.z as f32],
                }
            } else {
                [0.0_f32, 1.0, 0.0]
            };

            // UV（stream 缺失时用 dummy）
            let uv = if has_uv {
                let uv_idx = m.vertex_uv.indices[i] as usize;
                let t = unsafe { m.vertex_uv.values.data.add(uv_idx).read() };
                // FBX/ufbx 的 V 原点方向与 glTF/Cesium 相反，统一翻转 V。
                // 这是树叶上下颠倒、箱子/大门文字倒置的通用原因；不翻几何，只翻贴图坐标。
                [t.x as f32, 1.0 - t.y as f32]
            } else {
                [0.0_f32, 0.0]
            };

            let key = VKey::new(pos, nrm, uv);
            let vi = match dedup.get(&key) {
                Some(&existing) => existing,
                None => {
                    let new_idx = positions.len() as u32;
                    positions.push(pos);
                    normals.push(nrm);
                    uvs.push(uv);
                    dedup.insert(key, new_idx);
                    new_idx
                }
            };
            face_verts.push(vi);
        }

        // fan triangulation: n-gon → n-2 个三角 [v0, vi, vi+1]
        for i in 1..n_verts - 1 {
            let mut tri = [face_verts[0], face_verts[i], face_verts[i + 1]];
            if flip_winding {
                tri.swap(1, 2);
            }
            indices.extend_from_slice(&tri);
        }
    }

    if positions.is_empty() || indices.is_empty() {
        return None;
    }

    // 缺失 stream 的 dummy 已逐顶点写入，这里只需保证三者长度一致
    debug_assert_eq!(normals.len(), positions.len());
    debug_assert_eq!(uvs.len(), positions.len());

    Some(MeshData {
        positions,
        normals,
        uvs,
        indices,
        name,
        texture_uri,
        texture_bytes,
        texture_mime,
        texture_has_alpha,
        flip_winding,
        // 默认值；调用方在拿到 MeshData 后会用 FBX PBR 参数覆盖
        metallic_factor: 0.0,
        roughness_factor: 1.0,
        // pick 属性：全局构件索引由 extract_mesh_parts 填；batch_ids 由 tile 组装处按
        // 构件在 tile 内的局部序号逐顶点填充（merge 时随顶点段拼接）
        feature_gid: 0,
        batch_ids: Vec::new(),
    })
}

/// 构建 hierarchical tileset.json
///
/// 简化版：根节点直接挂所有 tile 的 children 数组（不考虑父子 LOD 层级）
/// 构建 hierarchical tileset.json（对齐 CesiumLab 格式）
///
/// 关键约定（参考 cesiumlab3 model2tiles 输出）：
/// 1. 每个 leaf tile 直接挂在 root 下（flat 结构，没有中间 LOD 节点）
/// 2. leaf `geometricError = 0`，root `geometricError = 整模型最大 extent`
/// 3. `refine: "REPLACE"` 在 root 上，Cesium 看到 GE=0 就知道这是叶子
/// 4. tile 文件名 `NoLod_<idx>.b3dm`，跟 cesiumlab 一致
/// 5. boundingVolume 用 `box`（12 元局部米坐标），不用 `region`
fn build_tileset(
    tree: &octree::OctreeNode,
    _tile_count: usize,
    tile_bounds: &[Option<([f32; 3], [f32; 3])>],
    _tile_sizes: &HashMap<usize, u64>,
    dead: &HashSet<usize>,
) -> serde_json::Value {
    let mut leaves = Vec::new();
    tree.collect_leaves(&mut leaves);
    leaves.retain(|l| !l.mesh_ids.is_empty() && !dead.contains(&(*l as *const octree::OctreeNode as usize)));

    let mut children = Vec::new();
    // root 的 AABB = 所有 tile 真实 AABB 的并集（保证 root 一定包住全部几何）
    let mut rmin = [f32::INFINITY; 3];
    let mut rmax = [f32::NEG_INFINITY; 3];

    for (i, leaf) in leaves.iter().enumerate() {
        // 优先用真实几何 AABB；拿不到（tile 被跳过）时回退到八叉树格子
        let (gmin, gmax) = match tile_bounds.get(i).copied().flatten() {
            Some(v) => v,
            None => (
                [leaf.bounds.min[0] as f32, leaf.bounds.min[1] as f32, leaf.bounds.min[2] as f32],
                [leaf.bounds.max[0] as f32, leaf.bounds.max[1] as f32, leaf.bounds.max[2] as f32],
            ),
        };

        for k in 0..3 {
            if gmin[k] < rmin[k] { rmin[k] = gmin[k]; }
            if gmax[k] > rmax[k] { rmax[k] = gmax[k]; }
        }

        // 给包围盒加 padding：边界上留余量，避免浮点误差让贴边的 tile 被视锥剔除误杀
        let box12 = aabb_minmax_to_box12_padded(gmin, gmax);

        // uri 不带 ?v=（查询串破坏浏览器磁盘缓存），直接用文件名
        let uri = format!("NoLod_{}.b3dm", i);

        children.push(serde_json::json!({
            "boundingVolume": { "box": box12 },
            "geometricError": 0,
            "refine": "REPLACE",
            "content": {
                "uri": uri
            }
        }));
    }

    let root_box = aabb_minmax_to_box12(rmin, rmax);
    let root_extent = ((rmax[0] - rmin[0])
        .max(rmax[1] - rmin[1])
        .max(rmax[2] - rmin[2])) as f64;

    serde_json::json!({
        "asset": {
            "generatetool": "RustFBX2Tiles",
            "version": "1.0"
        },
        "geometricError": root_extent,
        "refine": "REPLACE",
        "root": {
            "boundingVolume": { "box": root_box },
            "geometricError": root_extent,
            "refine": "REPLACE",
            "children": children
        }
    })
}

/// (min, max) 数组转 3D Tiles box，并预留 padding
///
/// padding = 最大边长 × 2% + 0.5m。留余量是为了避免恰好贴边的几何因浮点误差
/// 被视锥剔除误判（表现为"放大时模型局部消失，缩小后又出现"）。
/// 把 tile AABB 加上足够的 padding，让相邻 tile 自动重叠。
///
/// padding 设计目标：
/// 1. 与 tile 半轴长度挂钩：大 tile 加得多，小 tile 加得少。
/// 2. 保证任何相邻两个 tile 至少有"半轴长度的一半"重叠，防止缝。
/// 3. 最小 padding 1m，应对浮点误差。
///
/// 注意：这是包围盒**向外扩**，避免几何边缘被视锥剔除误剔，
/// 同时让相邻 tile 包围盒自然重叠（重叠区域的相机会被分配给任意一个 tile）。
fn aabb_minmax_to_box12_padded(min: [f32; 3], max: [f32; 3]) -> Vec<f32> {
    let hx = (max[0] - min[0]) * 0.5;
    let hy = (max[1] - min[1]) * 0.5;
    let hz = (max[2] - min[2]) * 0.5;
    // 每轴各加 50% 的半轴长度（即半轴加宽到 1.5 倍），让相邻 tile 至少有半轴长度的重叠
    let pad_x = hx * 0.5 + 1.0;
    let pad_y = hy * 0.5 + 1.0;
    let pad_z = hz * 0.5 + 1.0;
    let m0 = [min[0] - pad_x, min[1] - pad_y, min[2] - pad_z];
    let m1 = [max[0] + pad_x, max[1] + pad_y, max[2] + pad_z];
    aabb_minmax_to_box12(m0, m1)
}

/// (min, max) 数组转 3D Tiles box（12 元：centerXYZ + 3 个半轴向量）
///
/// 【关键】输入是 Y-up（glTF 坐标系）的 AABB，输出 box 必须是 tile 坐标系（Z-up）。
/// Cesium 视锥剔除按 tile 坐标系用 box，渲染 glTF 时做 Y-up→Z-up（(x,y,z)→(x,z,-y)）
/// 转换。box 若不预转换，与渲染后的几何错位 90°（Y/Z 互换）——
/// 这就是之前"放大时模型消失、缩小恢复"的根因。
///
/// 转换：center (cx,cy,cz)_yup → (cx, -cz, cy)_tiles
///       X 半轴不变；Y-up Y 半轴（高度）→ tiles Z 半轴；Y-up Z 半轴（北）→ tiles Y 半轴。
/// 用 CesiumLab mainview 实测数据验证过：其 box center (304.24,-96.80,0.19)
/// 恰等于 Y-up AABB 中心 (304.24, 0.19, 96.8) 的 (x,-z,y) 变换。
fn aabb_minmax_to_box12(min: [f32; 3], max: [f32; 3]) -> Vec<f32> {
    let cx = (min[0] + max[0]) * 0.5;
    let cy = (min[1] + max[1]) * 0.5;
    let cz = (min[2] + max[2]) * 0.5;
    let hx = (max[0] - min[0]) * 0.5;
    let hy = (max[1] - min[1]) * 0.5;
    let hz = (max[2] - min[2]) * 0.5;
    vec![
        cx, -cz, cy,           // center（tiles：X=东, Y=-yupZ, Z=yupY 高度）
        hx, 0.0, 0.0,          // X 半轴（东）
        0.0, hz, 0.0,          // tiles-Y 半轴（北，来自 yup Z）
        0.0, 0.0, hy,          // tiles-Z 半轴（高度，来自 yup Y）
    ]
}

/// AABB 转 3D Tiles box（12 元）——同样做 Y-up→Z-up 转换（见 aabb_minmax_to_box12 注释）
#[allow(dead_code)]
fn aabb_to_box12(bounds: &octree::Aabb) -> Vec<f32> {
    aabb_minmax_to_box12(
        [bounds.min[0] as f32, bounds.min[1] as f32, bounds.min[2] as f32],
        [bounds.max[0] as f32, bounds.max[1] as f32, bounds.max[2] as f32],
    )
}