#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""切片产物体检脚本：验证 b3dm/GLB 输出的结构完整性。

用法：
    python tools/validate_tiles.py <输出目录>

检查项（都是实际踩过的坑）：
1. b3dm 头部 + Feature/Batch Table JSON 可解析，BATCH_LENGTH 与 Batch Table 行数一致
2. GLB chunk 结构合法（JSON/BIN chunk 对齐、长度不越界）
3. 每个内嵌 image 的 bufferView 指向真实图片字节流（JPEG FFD8FF / PNG 89504E47）
   —— 历史坑：加 _BATCHID 时 geometry_view_count 没联动改，image 索引偏移指向几何数据
4. 每个 primitive 都有 _BATCHID attribute，且 batchId 值域 ⊂ [0, BATCH_LENGTH)
   —— 历史坑：material.baseColorTexture.index 误用 images 数组索引（应 textures 数组）
5. material 的 baseColorTexture.index ⊂ [0, textures 数组长度)
"""
import json
import struct
import sys
import glob
import os


def parse_b3dm(fp):
    raw = open(fp, "rb").read()
    if raw[:4] != b"b3dm":
        raise ValueError("不是 b3dm 文件")
    ver = struct.unpack_from("<I", raw, 4)[0]
    total_len = struct.unpack_from("<I", raw, 8)[0]
    ft_len = struct.unpack_from("<I", raw, 12)[0]
    bin_len = struct.unpack_from("<I", raw, 16)[0]
    bt_len = struct.unpack_from("<I", raw, 20)[0]
    bt_bin_len = struct.unpack_from("<I", raw, 24)[0]
    if total_len != len(raw):
        raise ValueError(f"b3dm 长度不一致: header={total_len} 实际={len(raw)}")
    ft = json.loads(raw[28:28 + ft_len])
    bt = json.loads(raw[28 + ft_len:28 + ft_len + bt_len]) if bt_len else {}
    glb = raw[28 + ft_len + bt_len:]
    return ver, ft, bt, glb


def parse_glb(glb):
    if glb[:4] != b"glTF":
        raise ValueError("GLB magic 错误")
    off = 12
    g, bin_start, bin_len = None, None, 0
    while off + 8 <= len(glb):
        clen, ctype = struct.unpack_from("<II", glb, off)
        data = glb[off + 8:off + 8 + clen]
        if ctype == 0x4E4F534A:
            g = json.loads(data)
        elif ctype == 0x004E4942:
            bin_start, bin_len = off + 8, clen
        off += 8 + clen + ((4 - clen % 4) % 4)
    if g is None or bin_start is None:
        raise ValueError("GLB 缺 JSON/BIN chunk")
    return g, bin_start, bin_len


def check_tile(fp):
    errors = []
    ver, ft, bt, glb = parse_b3dm(fp)
    if ver != 1:
        errors.append(f"b3dm version={ver}")
    batch_len = ft.get("BATCH_LENGTH", 0)

    g, bin_start, bin_len = parse_glb(glb)

    # 3. 图片字节流
    for i, img in enumerate(g.get("images", [])):
        bv_idx = img.get("bufferView")
        if bv_idx is None:
            continue
        if bv_idx >= len(g["bufferViews"]):
            errors.append(f"image[{i}] bufferView={bv_idx} 越界")
            continue
        bv = g["bufferViews"][bv_idx]
        d = glb[bin_start + bv["byteOffset"]: bin_start + bv["byteOffset"] + bv["byteLength"]]
        if not (d[:3] == b"\xff\xd8\xff" or d[:4] == b"\x89PNG"):
            errors.append(f"image[{i}] bufferView={bv_idx} 不是合法图片 (头={d[:4].hex()})")

    # 4. _BATCHID 值域（componentType 可能是 5126 FLOAT 或 5125 UINT）
    max_bid = -1
    for mi, mesh in enumerate(g.get("meshes", [])):
        for pi, prim in enumerate(mesh["primitives"]):
            bid_acc = prim["attributes"].get("_BATCHID")
            if bid_acc is None:
                errors.append(f"mesh[{mi}].primitive[{pi}] 缺 _BATCHID")
                continue
            acc = g["accessors"][bid_acc]
            bv = g["bufferViews"][acc["bufferView"]]
            start = bin_start + bv["byteOffset"]
            for k in range(acc["count"]):
                if acc["componentType"] == 5126:
                    v = struct.unpack_from("<f", glb, start + k * 4)[0]
                    v = int(round(v))
                else:
                    v = struct.unpack_from("<I", glb, start + k * 4)[0]
                if v > max_bid:
                    max_bid = v
    if batch_len > 0 and max_bid >= batch_len:
        errors.append(f"_BATCHID 最大值 {max_bid} 越界 (BATCH_LENGTH={batch_len})")

    # 5. material 贴图索引
    tex_count = len(g.get("textures", []))
    for mi, mat in enumerate(g.get("materials", [])):
        bct = mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if bct is not None and bct["index"] >= tex_count:
            errors.append(f"material[{mi}] baseColorTexture.index={bct['index']} 越界 (textures={tex_count})")

    # 6. indices 不得与 _BATCHID 共用 bufferView（历史坑：indices 重复键覆盖指向 batchId 数据，
    #    Cesium 读 batchId 当三角形索引 → 静默不渲染）
    for mi, mesh in enumerate(g.get("meshes", [])):
        for pi, prim in enumerate(mesh["primitives"]):
            bid = prim["attributes"].get("_BATCHID")
            idx = prim.get("indices")
            if bid is None or idx is None:
                continue
            if g["accessors"][bid]["bufferView"] == g["accessors"][idx]["bufferView"]:
                errors.append(f"mesh[{mi}].prim[{pi}] indices 与 _BATCHID 共用 bufferView（重复键 bug）")
            iv = g["bufferViews"][g["accessors"][idx]["bufferView"]]
            if iv.get("target") != 34963:
                errors.append(f"mesh[{mi}].prim[{pi}] indices bufferView target != 34963(ELEMENT_ARRAY)")

    # 1. Batch Table 行数一致性
    if "name" in bt and len(bt["name"]) != batch_len:
        errors.append(f"Batch Table name 行数 {len(bt['name'])} != BATCH_LENGTH {batch_len}")

    return errors


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    tiles = sorted(glob.glob(os.path.join(out_dir, "*.b3dm")))
    if not tiles:
        print(f"[体检] {out_dir} 下没有 b3dm 文件")
        return 1
    print(f"[体检] {len(tiles)} 个 tile ...")
    bad = 0
    for fp in tiles:
        try:
            errors = check_tile(fp)
        except Exception as e:
            errors = [f"解析失败: {e}"]
        if errors:
            bad += 1
            print(f"  [X] {os.path.basename(fp)}:")
            for e in errors:
                print(f"      - {e}")
    scenetree = os.path.join(out_dir, "scenetree.json")
    st_info = ""
    if os.path.exists(scenetree):
        js = json.load(open(scenetree, encoding="utf-8"))
        st_info = f"，scenetree 构件 {len(js['scenes'][0]['children'])} 个"
    if bad == 0:
        print(f"[体检] 全部通过 ({len(tiles)} tile{st_info})")
        return 0
    print(f"[体检] {bad}/{len(tiles)} 个 tile 有问题")
    return 2


if __name__ == "__main__":
    sys.exit(main())
