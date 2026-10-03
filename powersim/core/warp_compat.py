"""Let PhysGaussian's MPM solver (written for warp 0.10) run on warp 1.x by accepting its legacy
`owner` kwarg in warp.types.array."""

import warp as wp

_PATCHED_ATTR = "_powersim_owner_kwarg_patched"


def patch_warp_array_owner_kwarg() -> None:
    """Idempotent."""
    if getattr(wp.types.array.__init__, _PATCHED_ATTR, False):
        return

    original_init = wp.types.array.__init__

    def _init_accepting_legacy_owner_kwarg(self, *args, owner=None, **kwargs):
        # `owner` is accepted and discarded
        return original_init(self, *args, **kwargs)

    setattr(_init_accepting_legacy_owner_kwarg, _PATCHED_ATTR, True)
    wp.types.array.__init__ = _init_accepting_legacy_owner_kwarg
