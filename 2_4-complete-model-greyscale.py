import cvxpy as cp
import numpy as np
from PIL import Image, ImageDraw
from skimage import filters
from collections import defaultdict
import os
import random
import uuid

###############################
# Problem Dimensions
###############################

NUM_ROWS = 10
NUM_COLS = 10
BLOCK_SIZE = 8
NUM_COLORS = 10
SCALES = [1, 2, 4, 8]
EDGE_WEIGHT = 4.0
SIZE_BONUS = 2.0

random.seed(42)
np.random.seed(42)

###############################
# Polyomino definitions
###############################

class Polyomino:
    def __init__(self, name, blocks):
        self.name = name
        self.blocks = blocks
        self.height = max(r for r,c in blocks) + 1
        self.width  = max(c for r,c in blocks) + 1

    # rotates the polyomino 90 deg CW, retaining positive coordinates
    def rotate(self):
        # 90 deg CW rotation
        rotated = [(c, -r) for r,c in self.blocks]
        min_r = min(r for r,c in rotated)
        min_c = min(c for r,c in rotated)

        # translates to ensure positive coordinates
        rotated = [(r-min_r, c-min_c) for r,c in rotated]
        return Polyomino(self.name, rotated)

    # rotates by 0, 90, 180, 270 deg
    def rotations(self):
        rots = []
        s = self
        for _ in range(4):
            if not any(set(s.blocks) == set(r.blocks) for r in rots):
                rots.append(s)
            s = s.rotate()
        return rots

POLYOMINOES = [
    Polyomino("L",  [(0,0),(0,1),(1,0)]),
    Polyomino("I3", [(0,0),(0,1),(0,2)]),
    Polyomino("D2h",[(0,0),(0,1)]),
    Polyomino("D2v",[(0,0),(1,0)])
]

# Each polyomino gets assigned a single color. This makes things a little more interesting since the linear
# program now has to balance tiling + brightness/color-matching constraints!
color_to_polyomino = {
    c: random.choice(POLYOMINOES)
    for c in range(NUM_COLORS)
}

###############################
# Load target image 
###############################

TARGET_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'sources/frankenstein.png')
OUTPUT_IMAGE = f"output/edge-aware-v1-{uuid.uuid4().hex}.png"

img = Image.open(TARGET_IMAGE_PATH).convert("L") # converts to greyscale
img = img.resize((NUM_COLS*BLOCK_SIZE, NUM_ROWS*BLOCK_SIZE), Image.LANCZOS)
img_arr = np.array(img)

block_brightness = np.zeros((NUM_ROWS, NUM_COLS))
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        block = img_arr[
            i*BLOCK_SIZE:(i+1)*BLOCK_SIZE,
            j*BLOCK_SIZE:(j+1)*BLOCK_SIZE
        ]
        block_brightness[i,j] = round(block.mean() / 255.0 * (NUM_COLORS - 1))

# apply the Laplacian operator to the image to get an "edge map" (i.e pixels that are around edges 
# in the image get a high value, while pixels in a relatively flat region get a low value)
laplace_edges = filters.laplace(img_arr / 255.0)
edge_block = np.zeros((NUM_ROWS, NUM_COLS))
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        block = laplace_edges[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE,
                            j*BLOCK_SIZE:(j+1)*BLOCK_SIZE]
        # its very important to take the absolute value here! because the way the laplacian operator works 
        # is by detecting "zero-crossings", i.e rapid changes from negative to positive values
        # taking just the mean would produce a misleading edge map
        edge_block[i,j] = np.mean(np.abs(block))
# normalize mean laplacians to be in the range [0, 1]
edge_block /= np.max(edge_block)
edge_block = np.clip(edge_block, 0, 1)
print(edge_block)

###############################
# Generating tiles
###############################

def generate_tiles():
    tiles = []
    vals  = []
    for c in range(NUM_COLORS):
        v = int(c / (NUM_COLORS - 1) * 255)
        t = Image.new("L", (BLOCK_SIZE, BLOCK_SIZE), v)
        tiles.append(t)
        vals.append(v)
    vals = np.array(vals)
    return tiles, vals

colored_tiles, brightness_values = generate_tiles()

# Brings all brightness values in the discrete range [0, 9]
normalized_brightness = (brightness_values - brightness_values.min()) / \
                        (brightness_values.max() - brightness_values.min()) * (NUM_COLORS - 1)

###############################
# Placement generation
###############################

placements = []
block_to_placements = defaultdict(list)

# anchor == top-left block position
# the method just gives you all the blocks in footprint of this shape at this scale
def expanded_blocks(shape, scale, anchor):
    ai, aj = anchor
    blocks = []
    for dr, dc in shape.blocks:
        for u in range(scale):
            for v in range(scale):
                i = ai + dr*scale + u
                j = aj + dc*scale + v
                blocks.append((i,j))
    return blocks

for c in range(NUM_COLORS):
    base = color_to_polyomino[c]
    for shape in base.rotations():
        for S in SCALES:
            max_i = NUM_ROWS - shape.height*S
            max_j = NUM_COLS - shape.width*S
            for i in range(max_i + 1):
                for j in range(max_j + 1):
                    blocks = expanded_blocks(shape, S, (i,j))
                    p = len(placements) # used later to index into the current placement
                    placements.append((c, shape, S, (i,j), blocks))
                    for block in blocks:
                        block_to_placements[block].append(p)

NUM_PLACEMENTS = len(placements)
print("Total placements:", NUM_PLACEMENTS)

###############################
# CVXPY model
###############################

x = cp.Variable(NUM_PLACEMENTS, boolean=True)

constraints = []
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        constraints.append(cp.sum(x[block_to_placements[(i,j)]]) == 1)

###############################
# Objective function 
###############################

costs = np.zeros(NUM_PLACEMENTS)

for p, (c, shape, S, (i,j), blocks) in enumerate(placements):
    err = 0
    max_edge = 0
    for (ii,jj) in blocks:
        err += (normalized_brightness[c]-block_brightness[ii,jj])**2
        max_edge = max(max_edge, edge_block[ii,jj])

    err /= len(blocks)
    edge_pen = EDGE_WEIGHT * max_edge * (S-1)**2
    size_bonus = -SIZE_BONUS * (S-1)

    costs[p] = err + edge_pen + size_bonus

problem = cp.Problem(cp.Minimize(costs @ x), constraints)

###############################
# Solve it!
###############################

print("Solving...")
problem.solve(verbose=True)
print("Status:", problem.status)

###############################
# Render image 
###############################

result = Image.new("L",(NUM_COLS*BLOCK_SIZE,NUM_ROWS*BLOCK_SIZE),255)
draw = ImageDraw.Draw(result)

for p, val in enumerate(x.value):
    if val > 0.5:
        c, shape, S, (i,j), blocks = placements[p]
        block_set = set(blocks)

        for (ii,jj) in blocks:
            result.paste(
                colored_tiles[c],
                (jj*BLOCK_SIZE, ii*BLOCK_SIZE)
            )

        # draw a border 
        for (ii,jj) in blocks:
            x0 = jj * BLOCK_SIZE
            y0 = ii * BLOCK_SIZE
            x1 = x0 + BLOCK_SIZE
            y1 = y0 + BLOCK_SIZE

            # The idea here is that if the left neighbour is not in the block set, 
            # draw a border, if the top neighbour does not exist in the block set, draw 
            # a border, and so on...

            if (ii-1, jj) not in block_set:
                draw.line([(x0,y0),(x1,y0)], fill=0, width=2)
            
            if (ii+1, jj) not in block_set:
                draw.line([(x0,y1),(x1,y1)], fill=0, width=2)
            
            if (ii, jj-1) not in block_set:
                draw.line([(x0,y0),(x0,y1)], fill=0, width=2)
            
            if (ii, jj+1) not in block_set:
                draw.line([(x1,y0),(x1,y1)], fill=0, width=2)

result.save(OUTPUT_IMAGE)
print("Saved:", OUTPUT_IMAGE)
