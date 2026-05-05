#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import subprocess
import sys
import types
import traceback
import warnings

import torch
import numpy as np
from mmcv import Config
from mmcv.utils import import_modules_from_strings
from mmdet.models import build_detector


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OPEN_LOOP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
MOMAD_ORIG_ROOT = os.environ.get("MOMAD_ORIG_ROOT")


VARIANT_CONFIGS = {
    "momad": os.path.join(MOMAD_ORIG_ROOT, "open_loop/projects/configs/MomAD_small_stage2_roboAD_6s.py")
    if MOMAD_ORIG_ROOT else None,
    "momad_mlp_param": os.path.join(MOMAD_ORIG_ROOT, "open_loop/projects/configs/MomAD_small_stage2_roboAD_6s_mlp_param.py")
    if MOMAD_ORIG_ROOT else None,
    "mfpad": os.path.join(OPEN_LOOP_ROOT, "projects/configs/MomAD_small_stage2_roboAD_6s.py"),
    "mfpad_gru": os.path.join(OPEN_LOOP_ROOT, "projects/configs/MomAD_small_stage2_roboAD_6s_gru.py"),
    "mfpad_transonly": os.path.join(OPEN_LOOP_ROOT, "projects/configs/MomAD_small_stage2_roboAD_6s_transonly.py"),
    "womemory": os.path.join(OPEN_LOOP_ROOT, "projects/configs/MomAD_small_stage2_roboAD_6s_womemory.py"),
    "woforgetting": os.path.join(OPEN_LOOP_ROOT, "projects/configs/MomAD_small_stage2_roboAD_6s_woforgetting.py"),
}


def _find_repo_root(cfg_path):
    parts = cfg_path.split(os.sep)
    if "open_loop" in parts:
        idx = parts.index("open_loop")
        return os.sep.join(parts[: idx + 1])
    return os.path.dirname(cfg_path)


def _clear_project_modules():
    for name in list(sys.modules.keys()):
        if name == "projects" or name.startswith("projects."):
            del sys.modules[name]


def _patch_nuscenes():
    class DummyNuScenes:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, *args, **kwargs):
            return {"prev": ""}

    try:
        import nuscenes.nuscenes as nusc_mod
        nusc_mod.NuScenes = DummyNuScenes
    except Exception:
        mod = types.ModuleType("nuscenes")
        sub = types.ModuleType("nuscenes.nuscenes")
        sub.NuScenes = DummyNuScenes
        mod.nuscenes = sub
        sys.modules["nuscenes"] = mod
        sys.modules["nuscenes.nuscenes"] = sub


def _patch_deformable_ext():
    mod = types.ModuleType("projects.mmdet3d_plugin.ops.deformable_aggregation_ext")

    def _not_impl(*args, **kwargs):
        raise RuntimeError("deformable_aggregation_ext is disabled for complexity analysis")

    mod.deformable_aggregation_forward = _not_impl
    mod.deformable_aggregation_backward = _not_impl
    sys.modules["projects.mmdet3d_plugin.ops.deformable_aggregation_ext"] = mod


def _patch_numpy_load():
    orig_load = np.load

    def safe_load(path, *args, **kwargs):
        try:
            return orig_load(path, *args, **kwargs)
        except FileNotFoundError:
            base = os.path.basename(path)
            candidates = [os.path.join(OPEN_LOOP_ROOT, "data/kmeans")]
            if MOMAD_ORIG_ROOT:
                candidates.append(os.path.join(MOMAD_ORIG_ROOT, "open_loop/data/kmeans"))
            for root in candidates:
                alt = os.path.join(root, base)
                if os.path.exists(alt):
                    return orig_load(alt, *args, **kwargs)
            raise

    np.load = safe_load


def _import_plugin(cfg, cfg_path):
    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg["custom_imports"])

    if hasattr(cfg, "plugin") and cfg.plugin:
        plugin_dir = cfg.plugin_dir if hasattr(cfg, "plugin_dir") else os.path.dirname(cfg_path)
        module_dir = os.path.dirname(plugin_dir)
        module_dir = module_dir.split("/")
        module_path = module_dir[0]
        for part in module_dir[1:]:
            module_path = module_path + "." + part
        importlib.import_module(module_path)


def _disable_pretrained(cfg):
    if hasattr(cfg, "model") and isinstance(cfg.model, dict):
        img_backbone = cfg.model.get("img_backbone", None)
        if isinstance(img_backbone, dict) and img_backbone.get("pretrained", None):
            img_backbone["pretrained"] = None


def _disable_deformable(cfg):
    if hasattr(cfg, "model") and isinstance(cfg.model, dict):
        if "use_deformable_func" in cfg.model:
            cfg.model["use_deformable_func"] = False


def _build_model(cfg_path):
    repo_root = _find_repo_root(cfg_path)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    os.chdir(repo_root)

    _clear_project_modules()
    _patch_nuscenes()
    _patch_deformable_ext()
    _patch_numpy_load()

    cfg = Config.fromfile(cfg_path)
    _import_plugin(cfg, cfg_path)
    _disable_pretrained(cfg)
    _disable_deformable(cfg)

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    model.eval()
    return model, cfg


def _count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _format_m(n):
    return f"{n / 1e6:.3f}"


def _find_refine_layer(model):
    head = getattr(model, "head", None)
    if head is None:
        return None
    motion_plan_head = getattr(head, "motion_plan_head", None)
    if motion_plan_head is None:
        return None
    layers = getattr(motion_plan_head, "layers", None)
    if layers is None:
        return None
    for layer in layers:
        if layer.__class__.__name__.startswith("MotionPlanningRefinementModule"):
            return layer
    return None


def _try_full_flops(model, cfg, device):
    input_shape = getattr(cfg, "input_shape", None)
    if not input_shape or len(input_shape) != 2:
        return None, "full FLOPs: missing input_shape in config"

    h = input_shape[1]
    w = input_shape[0]
    num_cams = 6
    img = torch.randn(1, num_cams, 3, h, w, device=device)

    fvcore_err = None
    try:
        from fvcore.nn import FlopCountAnalysis
        try:
            flops = FlopCountAnalysis(model, (img,)).total()
            return flops, "full FLOPs: fvcore"
        except Exception as exc:
            fvcore_err = exc
    except Exception as exc:
        fvcore_err = exc

    try:
        from thop import profile
        flops, _params = profile(model, inputs=(img,), verbose=False)
        return flops, "full FLOPs: thop"
    except Exception as exc:
        return None, f"full FLOPs failed: fvcore={fvcore_err}; thop={exc}"


def _try_refine_flops(refine_layer, cfg, device):
    if refine_layer is None:
        return None, "refine FLOPs: refine layer not found"

    try:
        from fvcore.nn import FlopCountAnalysis
        use_fvcore = True
    except Exception as exc:
        use_fvcore = False
        fvcore_err = exc
    else:
        fvcore_err = None

    fut_ts = cfg.get("fut_ts", 12)
    fut_mode = cfg.get("fut_mode", 6)
    ego_fut_mode = cfg.get("ego_fut_mode", 6)
    embed_dims = cfg.get("embed_dims", 256)

    bs = 1
    num_anchor = 900
    token_count = 3 * ego_fut_mode

    motion_query = torch.randn(bs, num_anchor, fut_mode, embed_dims, device=device)
    plan_query = torch.randn(bs, 1, token_count, embed_dims, device=device)
    ego_feature = torch.randn(bs, 1, embed_dims, device=device)
    ego_anchor_embed = torch.randn(bs, 1, embed_dims, device=device)

    if use_fvcore:
        try:
            flops = FlopCountAnalysis(
                refine_layer,
                (motion_query, plan_query, ego_feature, ego_anchor_embed),
            ).total()
            return flops, "refine FLOPs: fvcore"
        except Exception as exc:
            fvcore_err = exc

    try:
        from thop import profile
        flops, _params = profile(
            refine_layer,
            inputs=(motion_query, plan_query, ego_feature, ego_anchor_embed),
            verbose=False,
        )
        return flops, "refine FLOPs: thop"
    except Exception as exc:
        return None, f"refine FLOPs failed: fvcore={fvcore_err}; thop={exc}"


def _summarize_variant(name, cfg_path):
    model, cfg = _build_model(cfg_path)
    total, trainable = _count_params(model)

    head = getattr(model, "head", None)
    motion_plan_head = getattr(head, "motion_plan_head", None) if head is not None else None
    head_total, head_train = (0, 0)
    if motion_plan_head is not None:
        head_total, head_train = _count_params(motion_plan_head)

    refine_layer = _find_refine_layer(model)
    refine_total, refine_train = (0, 0)
    if refine_layer is not None:
        refine_total, refine_train = _count_params(refine_layer)

    device = torch.device("cpu")
    model.to(device)
    if refine_layer is not None:
        refine_layer.to(device)

    full_flops, full_note = _try_full_flops(model, cfg, device)
    if full_flops is not None:
        flops_val = full_flops
        flops_note = full_note
        flops_scope = "full"
    else:
        refine_flops, refine_note = _try_refine_flops(refine_layer, cfg, device)
        flops_val = refine_flops
        if full_note:
            flops_note = f"{full_note}; {refine_note}"
        else:
            flops_note = refine_note
        flops_scope = "refine"

    return {
        "variant": name,
        "params_total": total,
        "params_trainable": trainable,
        "head_params_total": head_total,
        "head_params_trainable": head_train,
        "refine_params_total": refine_total,
        "refine_params_trainable": refine_train,
        "flops": flops_val,
        "flops_scope": flops_scope,
        "flops_note": flops_note,
    }


def _run_single(variant):
    cfg_path = VARIANT_CONFIGS[variant]
    if cfg_path is None:
        return {
            "variant": variant,
            "error": "set MOMAD_ORIG_ROOT to evaluate this variant",
        }
    try:
        return _summarize_variant(variant, cfg_path)
    except Exception as exc:
        err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return {"variant": variant, "error": err}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=VARIANT_CONFIGS.keys(),
        help="variant name; if omitted, run all",
    )
    parser.add_argument("--json", action="store_true", help="print json result for single variant")
    args = parser.parse_args()

    if args.json:
        warnings.filterwarnings("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"
        if not args.variant:
            raise SystemExit("--json requires --variant")
        print(json.dumps(_run_single(args.variant)))
        return

    variants = [args.variant] if args.variant else list(VARIANT_CONFIGS.keys())
    results = []
    if args.variant:
        results.append(_run_single(args.variant))
    else:
        for name in variants:
            proc = subprocess.run(
                [sys.executable, __file__, "--variant", name, "--json"],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                results.append({"variant": name, "error": proc.stderr.strip() or proc.stdout.strip()})
                continue
            out_lines = [line for line in proc.stdout.splitlines() if line.strip()]
            json_line = None
            for line in reversed(out_lines):
                if line.strip().startswith("{") and line.strip().endswith("}"):
                    json_line = line.strip()
                    break
            if not json_line:
                results.append({"variant": name, "error": "json parse failed: no json line found"})
                continue
            try:
                results.append(json.loads(json_line))
            except Exception as exc:
                results.append({"variant": name, "error": f"json parse failed: {exc}"})

    print("| Variant | Params (M) | Trainable Params (M) | FLOPs (G) | Notes |")
    print("|--------|------------|----------------------|-----------|-------|")
    for r in results:
        if "error" in r:
            print(f"| {r['variant']} | - | - | - | build failed: {r['error']} |")
            continue
        flops_g = "-"
        note = r["flops_note"]
        if r["flops"] is not None:
            flops_g = f"{r['flops'] / 1e9:.3f}"
            if r["flops_scope"] == "refine":
                note = f"{note}; refine-only"
        print(
            "| {variant} | {params_total} | {params_trainable} | {flops} | {note} |".format(
                variant=r["variant"],
                params_total=_format_m(r["params_total"]),
                params_trainable=_format_m(r["params_trainable"]),
                flops=flops_g,
                note=note,
            )
        )

    print("\nPlanning head params (M):")
    for r in results:
        if "error" in r:
            continue
        print(
            "- {variant}: total={total}, trainable={trainable}, refine_total={ref_total}, refine_trainable={ref_train}".format(
                variant=r["variant"],
                total=_format_m(r["head_params_total"]),
                trainable=_format_m(r["head_params_trainable"]),
                ref_total=_format_m(r["refine_params_total"]),
                ref_train=_format_m(r["refine_params_trainable"]),
            )
        )


if __name__ == "__main__":
    main()
