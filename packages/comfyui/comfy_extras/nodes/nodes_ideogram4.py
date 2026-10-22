"""Ideogram 4 sampling helper
"""

from typing_extensions import override

from comfy.ldm.ideogram4.sampling import ideogram4_sigmas
from comfy_api.latest import ComfyExtension, io


class Ideogram4Scheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Ideogram4Scheduler",
            display_name="Ideogram 4 Scheduler",
            category="model/sampling/schedulers",
            inputs=[
                # ideogram4 sampler_configs.py V4_QUALITY_48, the default preset: 48 steps, mu 0, std 1.5
                io.Int.Input("steps", default=48, min=1, max=200),
                io.Int.Input("width", default=1024, min=256, max=8192, step=16),
                io.Int.Input("height", default=1024, min=256, max=8192, step=16),
                io.Float.Input("mu", default=0.0, min=-10.0, max=10.0, step=0.05),
                io.Float.Input("std", default=1.5, min=0.1, max=5.0, step=0.05),
                io.Boolean.Input("full_denoise", default=False, optional=True,
                                 tooltip="End at sigma 0 instead of where the reference loop stops (1 - t_max, about 5.5e-4)."),
            ],
            outputs=[io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, steps, width, height, mu, std, full_denoise=False) -> io.NodeOutput:
        sigmas = ideogram4_sigmas(steps, width, height, mu, std)
        if full_denoise:
            sigmas[-1] = 0.0
        return io.NodeOutput(sigmas)


class Ideogram4Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Ideogram4Scheduler]


async def comfy_entrypoint() -> Ideogram4Extension:
    return Ideogram4Extension()
