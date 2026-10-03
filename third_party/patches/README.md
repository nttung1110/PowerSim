# Patches applied to third_party/powerfoam

`third_party/powerfoam` is PowerFoam's source tree (Apache-2.0) with two local changes already
applied. The diffs are kept here for reference and for re-applying to a newer upstream:

| Patch | What | Needed by |
|---|---|---|
| `powerfoam_rasterize_nonpinhole.patch` | `count_visible_kernel` / `write_visible_kernel` get the cone-tile arguments when `is_pinhole` is false, so the tile culling in `Rasterizer` works for non-pinhole COLMAP cameras (the kernels already took them; the launches did not pass them). | the telephone scene (`telephone_sfm@2ed8d7fa`, `is_pinhole: false`) |
| `powerfoam_render_orbit.patch` | `render_orbit.py` derives orbit defaults (azimuth range, radius, fov) from the training cameras and accepts them as flags. | orbit videos of composited scenes (foam_edit) |

Base: the PowerFoam revision vendored in the research repo (git 30e7d21847a968614244514b5d788e4d8a192776 of PhysFoam).
Apply to a clean upstream checkout with `git apply third_party/patches/*.patch` from the repo root.
