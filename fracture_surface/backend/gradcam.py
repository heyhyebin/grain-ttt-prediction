import io
import base64

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def make_gradcampp_base64(model, input_tensor, original_image, target_class_idx, img_size=224):
    model.eval()

    gradients = []
    activations = []

    target_layer = model.aspp

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    score = output[:, target_class_idx]

    model.zero_grad()
    score.backward()

    acts = activations[0].detach()
    grads = gradients[0].detach()

    grads_power_2 = grads ** 2
    grads_power_3 = grads ** 3

    sum_acts = torch.sum(acts, dim=(2, 3), keepdim=True)

    eps = 1e-8
    alpha = grads_power_2 / (
        2 * grads_power_2 + sum_acts * grads_power_3 + eps
    )

    positive_grads = F.relu(grads)
    weights = torch.sum(alpha * positive_grads, dim=(2, 3), keepdim=True)

    cam = torch.sum(weights * acts, dim=1).squeeze()
    cam = F.relu(cam)

    cam = cam.cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    img_np = np.array(original_image.resize((img_size, img_size)))
    heatmap = cv2.resize(cam, (img_size, img_size))

    mask = (heatmap > 0.45).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    overlay = img_np.copy()

    cv2.drawContours(
        overlay,
        contours,
        -1,
        color=(255, 70, 70),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    result = cv2.addWeighted(img_np, 0.82, overlay, 0.18, 0)

    result_img = Image.fromarray(result)

    buffer = io.BytesIO()
    result_img.save(buffer, format="PNG")

    forward_handle.remove()
    backward_handle.remove()

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"