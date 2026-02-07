import os
import cvxpy as cp
import numpy as np
import uuid
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage import filters
import json

###############################
# Problem Dimensions 
###############################

NUM_ROWS = 30 
NUM_COLS = 30 
TILE_SIZES = [1, 2, 4, 8]
BLOCK_SIZE = 8
EDGE_WEIGHT = 4.0
TILE_BONUS = 2.0

###############################
# Generating tiles 
###############################

PALETTE_CONFIG = os.path.join(os.path.dirname(__file__), "colors/a-girl-with-pearl-earrings-colors.json")

with open(PALETTE_CONFIG, "r") as f:
    palette_data = json.load(f)

palette = [tuple(color) for color in palette_data["colors"]]
NUM_COLORS = len(palette)

def generate_color_tiles():
    tiles = []
    brightness_values = []

    for (r, g, b) in palette:
        tile_array = np.zeros((BLOCK_SIZE*max(TILE_SIZES), BLOCK_SIZE*max(TILE_SIZES), 3), dtype=np.uint8)
        tile_array[:] = [r, g, b]
        tile = Image.fromarray(tile_array, mode="RGB")
        tiles.append(tile)

        brightness = 0.299*r + 0.587*g + 0.114*b
        brightness_values.append(brightness)

    return tiles, np.array(brightness_values)

colored_tiles, brightness_values = generate_color_tiles()

# Brings all brightness values in the discrete range [0, 9]
normalized_brightness = (brightness_values - brightness_values.min()) / \
                        (brightness_values.max() - brightness_values.min()) * 9

###############################
# Load target image 
###############################

TARGET_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'sources/pearl_earring.jpg')
OUTPUT_IMAGE = f"output/edge-aware-v0-{uuid.uuid4().hex}.png"

greyscale_img = Image.open(TARGET_IMAGE_PATH).convert('L')
img = greyscale_img.resize((NUM_COLS*BLOCK_SIZE, NUM_ROWS*BLOCK_SIZE), Image.LANCZOS)
img_array = np.array(img)

# Compute brightness for each block
block_brightness = np.zeros((NUM_ROWS, NUM_COLS))
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        block = img_array[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE,
                          j*BLOCK_SIZE:(j+1)*BLOCK_SIZE]
        # again, bring the brightness value in the discrete range [0, 9]
        block_brightness[i,j] = round(block.mean() / 255.0 * 9)

# apply the Laplacian operator to the image to get an "edge map" (i.e pixels that are around edges 
# in the image get a high value, while pixels in a relatively flat region get a low value)
laplace_edges = filters.laplace(img_array/255.0)
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
edge_block /= np.percentile(edge_block, 95)
edge_block = np.clip(edge_block, 0, 1)
print(edge_block)

###############################
# Build cvxpy model 
###############################

print(f"Building model with {NUM_ROWS}x{NUM_COLS} = {NUM_ROWS*NUM_COLS} blocks...")

# Decision variables: x[c, i, j, s] ∈ {0,1} if color c at block (i,j) with size s
x = cp.Variable((NUM_COLORS, NUM_ROWS, NUM_COLS, len(TILE_SIZES)), boolean=True)
constraints = []

# Each block is covered exactly once
print("Adding coverage constraints...")
for i in range(NUM_ROWS):
    if i % 10 == 0:
        print(f"  Processing row {i}/{NUM_ROWS}...")
    for j in range(NUM_COLS):
        cover_blocks = []
        for c in range(NUM_COLORS):
            for s_idx, s in enumerate(TILE_SIZES):
                # check all possible top-left positions that could cover block (i,j)
                # in other words, any (i', j') is included in the footprint of block with top-left
                # position (i, j) iff i <= i' <= i + s and j <= j' <= j + s
                for ti in range(max(0, i-s+1), i+1): # +1 is courtesy of python's 0-indexing
                    for tj in range(max(0, j-s+1), j+1):
                        if ti+s <= NUM_ROWS and tj+s <= NUM_COLS:
                            cover_blocks.append(x[c,ti,tj,s_idx])
        constraints.append(cp.sum(cover_blocks) == 1)

print(f"Total constraints: {len(constraints)}")

###############################
# Objective function 
###############################

print("Building objective function...")
objective_terms = []

for c in range(NUM_COLORS):
    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):
            for s_idx, s in enumerate(TILE_SIZES):
                if i+s <= NUM_ROWS and j+s <= NUM_COLS:
                    # average the brightness error - normalized per block (to be [0, 9])
                    brightness_error = 0
                    edge_penalty = 0
                    max_edge_in_tile = 0 # largest laplacian value in the footprint of this tile
                    
                    for di in range(s):
                        for dj in range(s):
                            ii, jj = i+di, j+dj
                            brightness_error += (normalized_brightness[c]-block_brightness[ii,jj])**2
                            max_edge_in_tile = max(max_edge_in_tile, edge_block[ii,jj])
                    
                    avg_brightness_error = brightness_error / (s * s)

                    # strongly penalize large tiles on high-edge areas
                    edge_penalty = EDGE_WEIGHT * max_edge_in_tile * (s - 1) ** 2
                    
                    # reward larger tiles to encourage consolidation in smooth areas
                    tile_size_bonus = -TILE_BONUS * (s - 1)
                    
                    cost = avg_brightness_error + edge_penalty + tile_size_bonus
                    objective_terms.append(cost * x[c,i,j,s_idx])

objective = cp.Minimize(cp.sum(objective_terms))
problem = cp.Problem(objective, constraints)

###############################
# Solve it! 
###############################

print("\n" + "="*60)
print("Solving...")
print("="*60)
problem.solve(verbose=True)

print(f"\nSolver status: {problem.status}")
print(f"Optimal value: {problem.value}")

###############################
# Render image
###############################

print("\nBuilding mosaic image...")
mosaic_image = Image.new('RGB', (NUM_COLS*BLOCK_SIZE, NUM_ROWS*BLOCK_SIZE), color=(255,255,255))
solution = x.value

# Track which blocks have been placed to avoid overwriting
placed_blocks = np.zeros((NUM_ROWS, NUM_COLS), dtype=bool)

for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        for c in range(NUM_COLORS):
            for s_idx, s in enumerate(TILE_SIZES):
                # checking > 0.5 to avoid floating point equality issues (i.e 1 != 1.0000001)
                if i+s <= NUM_ROWS and j+s <= NUM_COLS and solution[c,i,j,s_idx] > 0.5:
                    # Check if this is a top-left corner that hasn't been placed
                    if not placed_blocks[i,j]:
                        tile = colored_tiles[c].resize((s*BLOCK_SIZE, s*BLOCK_SIZE), Image.NEAREST)
                        
                        # draw a black border around the tile 
                        draw = ImageDraw.Draw(tile)
                        draw.rectangle([0,0,tile.size[0]-1,tile.size[1]-1], outline=(0,0,0), width=1)
                        mosaic_image.paste(tile, (j*BLOCK_SIZE, i*BLOCK_SIZE))
                        
                        # Mark all blocks covered by this tile as placed
                        for di in range(s):
                            for dj in range(s):
                                placed_blocks[i+di, j+dj] = True

# fill any unplaced blocks (fallback) -- this should never happen!
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        if not placed_blocks[i,j]:
            mosaic_image.paste(Image.new('RGB', (BLOCK_SIZE,BLOCK_SIZE), (255,255,255)),
                               (j*BLOCK_SIZE, i*BLOCK_SIZE))

mosaic_image.save(OUTPUT_IMAGE)
print(f"\nSaved mosaic with edge-aware tile sizes: {OUTPUT_IMAGE}")

# Print statistics about tile sizes used
tile_size_counts = {s: 0 for s in TILE_SIZES}
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        for c in range(NUM_COLORS):
            for s_idx, s in enumerate(TILE_SIZES):
                if i+s <= NUM_ROWS and j+s <= NUM_COLS and solution[c,i,j,s_idx] > 0.5:
                    tile_size_counts[s] += 1

print("\n" + "="*60)
print("TILE SIZE DISTRIBUTION:")
print("="*60)
for s in TILE_SIZES:
    blocks_covered = tile_size_counts[s] * s * s
    percentage = (blocks_covered / (NUM_ROWS * NUM_COLS)) * 100
    print(f"  {s}x{s} tiles: {tile_size_counts[s]:4d} tiles covering {blocks_covered:5d} blocks ({percentage:5.1f}%)")
print(f"\nTotal blocks: {NUM_ROWS * NUM_COLS}")
print(f"Image dimensions: {NUM_COLS*BLOCK_SIZE}x{NUM_ROWS*BLOCK_SIZE} pixels")
