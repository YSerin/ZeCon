from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Union
from PIL.Image import Image


def show_edited_masked_image(
    title: str,
    source_image: Image,
    edited_image: Image,
    mask: Optional[Image] = None,
    path: Optional[Union[str, Path]] = None,
    distance: Optional[str] = None,
):
    fig_idx = 1
    rows = 1
    cols = 3 if mask is not None else 2

    fig = plt.figure(figsize=(12, 5))
    figure_title = f'Prompt: "{title}"'
    if distance is not None:
        figure_title += f" ({distance})"
    plt.title(figure_title)
    plt.axis("off")

    fig.add_subplot(rows, cols, fig_idx)
    fig_idx += 1
    _set_image_plot_name("Source Image")
    plt.imshow(source_image)

    if mask is not None:
        fig.add_subplot(rows, cols, fig_idx)
        _set_image_plot_name("Mask")
        plt.imshow(mask)
        plt.gray()
        fig_idx += 1

    fig.add_subplot(rows, cols, fig_idx)
    _set_image_plot_name("Edited Image")
    plt.imshow(edited_image)

    if path is not None:
        plt.savefig(path, bbox_inches="tight")
    else:
        plt.show(block=True)

    plt.close()


def _set_image_plot_name(name):
    plt.title(name)
    plt.xticks([])
    plt.yticks([])
