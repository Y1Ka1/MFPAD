import os

_base_path = os.path.join(os.path.dirname(__file__), "MomAD_small_stage2_roboAD_6s.py")
with open(_base_path, "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), _base_path, "exec"))

model["head"]["motion_plan_head"]["refine_layer"]["type"] = "MotionPlanningRefinementModuleGRU"
model["head"]["motion_plan_head"]["refine_layer"]["inst_dim"] = embed_dims
model["head"]["motion_plan_head"]["refine_layer"]["dropout_forget"] = 0.2
model["head"]["motion_plan_head"]["refine_layer"]["ins_k"] = 32
