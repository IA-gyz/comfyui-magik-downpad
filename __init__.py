from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .magik_downpad import MagikDownPad


class MagikDownPadExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MagikDownPad]


async def comfy_entrypoint() -> MagikDownPadExtension:
    return MagikDownPadExtension()
