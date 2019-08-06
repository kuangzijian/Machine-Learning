import matplotlib.pyplot as plt
import numpy as np


def displayData(X, tile_width=-1, padding=0):

    m, n = X.shape

    if tile_width < 0:
        tile_width = int(np.round(np.sqrt(n)))
    tile_height = n / tile_width

    display_rows = int(np.floor(np.sqrt(m)))
    display_columns = int(np.ceil(m / display_rows))

    tile_height_padded = int(tile_height) + padding * 2
    tile_width_padded = int(tile_width) + padding * 2
    data = np.zeros((display_rows * tile_height_padded, display_columns * tile_width_padded))

    for i in range(display_rows):
        for j in range(display_columns):
            tile = formatTile(X[i * display_rows + j,], tile_width, padding)
            tile = tile.T
            data[i * tile_height_padded:(i + 1) * tile_height_padded,
                 j * tile_width_padded:(j + 1) * tile_width_padded] = tile

    plt.imshow(data, cmap='gray', extent=[0, 1, 0, 1])


def formatTile(x, width=-1, padding=0):

    if width < 0:
        width = int(np.round(np.sqrt(len(x))))
    height = len(x) / width

    tile = np.ones((int(height) + padding * 2, width + padding * 2))

    for i in range(padding, int(height) + padding):
        tile[i, padding:(padding + width)] = x[((i - padding) * width):((i - padding) * width + width)]

    return tile