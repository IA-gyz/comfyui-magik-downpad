import torch
import torch.nn.functional as F
from comfy_api.latest import io


class MagikDownPad(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Magik.DownPad",
            display_name="Magik DownPad \U0001FA84",
            category="Magik/Image/Process",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input(
                    "downscale",
                    default=0.90,
                    min=0.00,
                    max=1.00,
                    step=0.01,
                    display_mode=io.NumberDisplay.number,
                ),
                # Dropdown list with powers of 2
                io.Combo.Input(
                    "multiple_of",
                    options=["1", "2", "4", "8", "16", "32", "64"],
                    default="64",
                ),
            ],
            outputs=[
                io.Image.Output("IMAGE", display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, image, downscale, multiple_of):
        # Convert the dropdown string selection back to an integer
        multiple_of = int(multiple_of)

        # image shape in ComfyUI is [batch_size, height, width, channels]
        b, h, w, c = image.shape

        # 1. Calculate target dimensions based on downscale factor
        target_w = int(w * downscale)
        target_h = int(h * downscale)

        # 2. Force the dimensions to be a multiple of 'multiple_of'
        new_w = (target_w // multiple_of) * multiple_of
        new_h = (target_h // multiple_of) * multiple_of

        # Failsafe: Ensure it's at least 1 pixel and doesn't exceed original dimensions
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        if new_w > w:
            new_w = max(1, (w // multiple_of) * multiple_of)
        if new_h > h:
            new_h = max(1, (h // multiple_of) * multiple_of)

        # 3. Permute image to [B, C, H, W] for PyTorch interpolation
        img_permuted = image.permute(0, 3, 1, 2)

        # 4. Resize the image using bilinear interpolation
        resized_img = F.interpolate(
            img_permuted,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )

        # Permute back to [B, H, W, C]
        resized_img = resized_img.permute(0, 2, 3, 1)

        # 5. Create a black canvas identical to the original image shape
        canvas = torch.zeros_like(image)

        # 6. Calculate top and left offsets to center the downscaled image
        pad_top = (h - new_h) // 2
        pad_left = (w - new_w) // 2

        # 7. Paste the resized image onto the center of the black canvas
        canvas[:, pad_top:pad_top+new_h, pad_left:pad_left+new_w, :] = resized_img

        return io.NodeOutput(canvas)
