import cvxpy as cp
import numpy as np
from PIL import Image, ImageDraw
from skimage import filters
from collections import defaultdict
import os
import random
import uuid
import json

###############################
# Problem Dimensions
###############################

NUM_ROWS = 20
NUM_COLS = 10
BLOCK_SIZE = 8
SCALES = [1, 2, 4, 8]
EDGE_WEIGHT = 5.0
SIZE_BONUS = 2.0

random.seed(42)
np.random.seed(42)

###############################
# Polyomino definitions
###############################

class Polyomino:
    def __init__(self, name, cells):
        self.name = name
        self.cells = cells
        self.height = max(r for r,c in cells) + 1
        self.width  = max(c for r,c in cells) + 1

    def rotate(self):
        rotated = [(c, -r) for r,c in self.cells]
        min_r = min(r for r,c in rotated)
        min_c = min(c for r,c in rotated)
        rotated = [(r-min_r, c-min_c) for r,c in rotated]
        return Polyomino(self.name, rotated)

    def rotations(self):
        rots = []
        s = self
        for _ in range(4):
            if not any(set(s.cells) == set(r.cells) for r in rots):
                rots.append(s)
            s = s.rotate()
        return rots

POLYOMINOES = [
    Polyomino("L",  [(0,0),(0,1),(1,0)]),
]

###############################
# Load palette config
###############################

PALETTE_CONFIG = os.path.join(os.path.dirname(__file__), "colors/mona-lisa-colors.json")

with open(PALETTE_CONFIG, "r") as f:
    palette_data = json.load(f)

palette = [tuple(color) for color in palette_data["colors"]]
NUM_COLORS = len(palette)

# each color gets assigned a single shape -- i.e not all shapes are available in all possible colors
color_to_polyomino = {
    c: POLYOMINOES[c % len(POLYOMINOES)]
    for c in range(NUM_COLORS)
}

###############################
# Load target image 
###############################

TARGET_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'sources/mona-lisa.jpg')
OUTPUT_IMAGE = f"output/edge-aware-v1-monalisa-{uuid.uuid4().hex}.png"

img = Image.open(TARGET_IMAGE_PATH).convert("L")
img = img.resize((NUM_COLS*BLOCK_SIZE, NUM_ROWS*BLOCK_SIZE), Image.LANCZOS)
img_arr = np.array(img)

block_brightness = np.zeros((NUM_ROWS, NUM_COLS))
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        block = img_arr[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE,
                        j*BLOCK_SIZE:(j+1)*BLOCK_SIZE]
        block_brightness[i,j] = round(block.mean() / 255.0 * 9)

###############################
# Edge map
###############################

laplace_edges = filters.laplace(img_arr / 255.0)

edge_block = np.zeros((NUM_ROWS, NUM_COLS))
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        block = laplace_edges[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE,
                              j*BLOCK_SIZE:(j+1)*BLOCK_SIZE]
        edge_block[i,j] = np.mean(np.abs(block))

edge_block /= np.percentile(edge_block, 95)
edge_block = np.clip(edge_block, 0, 1)

###############################
# Mona Lisa Palette (same as first script)
###############################

def generate_mona_lisa_palette():
    tiles = []
    brightness_vals = []

    for (r,g,b) in palette:
        tile = Image.new("RGB", (BLOCK_SIZE, BLOCK_SIZE), (r,g,b))
        tiles.append(tile)

        brightness = 0.299*r + 0.587*g + 0.114*b
        brightness_vals.append(brightness)

    brightness_vals = np.array(brightness_vals)
    return tiles, brightness_vals

colored_tiles, brightness_values = generate_mona_lisa_palette()

normalized_brightness = (brightness_values - brightness_values.min()) / \
                        (brightness_values.max() - brightness_values.min()) * 9

###############################
# Placement generation
###############################

placements = []
cell_to_placements = defaultdict(list)

def expanded_cells(shape, scale, anchor):
    ai, aj = anchor
    cells = []
    for dr, dc in shape.cells:
        for u in range(scale):
            for v in range(scale):
                i = ai + dr*scale + u
                j = aj + dc*scale + v
                cells.append((i,j))
    return cells

for c in range(NUM_COLORS):
    base = color_to_polyomino[c]
    for shape in base.rotations():
        for S in SCALES:
            max_i = NUM_ROWS - shape.height*S
            max_j = NUM_COLS - shape.width*S
            for i in range(max_i + 1):
                for j in range(max_j + 1):
                    cells = expanded_cells(shape, S, (i,j))
                    p = len(placements)
                    placements.append((c, shape, S, (i,j), cells))
                    for cell in cells:
                        cell_to_placements[cell].append(p)

NUM_PLACEMENTS = len(placements)
print("Total placements:", NUM_PLACEMENTS)

###############################
# CVXPY model
###############################

x = cp.Variable(NUM_PLACEMENTS, boolean=True)

constraints = []
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        constraints.append(cp.sum(x[cell_to_placements[(i,j)]]) == 1)

###############################
# Objective
###############################

costs = np.zeros(NUM_PLACEMENTS)

for p, (c, shape, S, (i,j), cells) in enumerate(placements):
    err = 0
    max_edge = 0
    for (ii,jj) in cells:
        err += (normalized_brightness[c]-block_brightness[ii,jj])**2
        max_edge = max(max_edge, edge_block[ii,jj])

    err /= len(cells)
    edge_pen = EDGE_WEIGHT * max_edge * (S-1)**2
    size_bonus = -SIZE_BONUS * (S-1)

    costs[p] = err + edge_pen + size_bonus

problem = cp.Problem(cp.Minimize(costs @ x), constraints)

###############################
# Solve
###############################

print("Solving...")
problem.solve(verbose=True)
print("Status:", problem.status)

###############################
# Render image
###############################

result = Image.new("RGB",(NUM_COLS*BLOCK_SIZE,NUM_ROWS*BLOCK_SIZE),(255,255,255))
draw = ImageDraw.Draw(result)

for p, val in enumerate(x.value):
    if val > 0.5:
        c, shape, S, (i,j), cells = placements[p]
        cell_set = set(cells)

        for (ii,jj) in cells:
            result.paste(
                colored_tiles[c],
                (jj*BLOCK_SIZE, ii*BLOCK_SIZE)
            )

        # borders
        for (ii,jj) in cells:
            x0 = jj * BLOCK_SIZE
            y0 = ii * BLOCK_SIZE
            x1 = x0 + BLOCK_SIZE
            y1 = y0 + BLOCK_SIZE

            if (ii-1, jj) not in cell_set:
                draw.line([(x0,y0),(x1,y0)], fill=(0,0,0), width=2)
            if (ii+1, jj) not in cell_set:
                draw.line([(x0,y1),(x1,y1)], fill=(0,0,0), width=2)
            if (ii, jj-1) not in cell_set:
                draw.line([(x0,y0),(x0,y1)], fill=(0,0,0), width=2)
            if (ii, jj+1) not in cell_set:
                draw.line([(x1,y0),(x1,y1)], fill=(0,0,0), width=2)

result.save(OUTPUT_IMAGE)
print("Saved:", OUTPUT_IMAGE)
