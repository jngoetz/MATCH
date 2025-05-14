# Using MATCH
## Changelog
### 2024-05-23
- Removed debug print lines
- Fix bug where output directory wouldn't be created if not already existing
- Fix bug determining color clusters - function max_L was returning an Luv with a nested Luv as the L value
- Correct required modules in this readme

### 2024-05-16
Initial upload

## Required Modules
Requires `pandas`, `numpy`, `colormath`, `scikit-learn`, and `scikit-image` modules.

To install via `pip` use:
```
python -m pip install pandas numpy colormath scikit-learn scikit-image
```

## Basic usage
To get the help message below, run (from the same folder as MATCH.py):
```
python MATCH.py --help
```

To run MATCH against a folder with images in it (eg. `images` in the same folder as MATCH.py) with parameters _percent pixel coverage_ and _filter color_, use the following command:
```
python MATCH.py --pct-pix-cov 0.7 --filter-color primary images
```

After running, the result should look like:
```
output/
├── images/
│  ├── img1/
│  │  ├── data.json     # contains metadata about the 'img1'
│  │  ├── target.png    # the 'img1' image, resized to the specified size
│  │  ├── primary1.png  # 'primary' is from the --filter-color parameter, 1 is the closest match
│  │  ├── primary2.png
│  │  └── primary3.png
│  └── img2/
├── candidates.csv      # contains the same as 'images.csv' but with the 'candidate' column added (True/False)
└── neighbors.csv       # contains all the candidates with their closest neighbors and distances
```


## MATCH command
Most parameters have defaults, so pay attention to those. You most likely only need the `dirs`, `--pct-pix-cov` and `--filter-color` parameters.
```
usage: MATCH.py [-h] [--img-max-size IMG_MAX_SIZE] [--images IMAGES]
                [--candidates CANDIDATES] [--candidate-only]
                [--filter-color FILTER_COLOR] [--pct-pix-cov PCT_PIX_COV]
                [--matching-type {primary,secondary,tertiary}]
                [--n-matching N_MATCHING] [--output-dir OUTPUT_DIR]
                [dirs ...]

positional arguments:
  dirs                  Directories to search for images, if the images csv is
                        not found (default: None)

options:
  -h, --help            show this help message and exit
  --img-max-size IMG_MAX_SIZE
                        Resize images to this standard size. For non-square
                        images, the maximum dimension is used. (default: 160)
  --images IMAGES       Path to save/load images and their color clusters.
                        This takes long to compute so it is recommended to
                        reuse one. (default: images.csv)
  --candidates CANDIDATES
                        Path to save/load candidates csv.If you wish to apply
                        filters on the candidates before generating the
                        matching images, pair this with '--candidate-only'
                        first to save this file and stop execution. After
                        making changes, run it again without '--candidate-
                        only' and it will use the modified candidates.
                        (default: None)
  --candidate-only      Only generate the candidates csv and stop execution
                        (default: False)
  --filter-color FILTER_COLOR
                        Which color to filter candidates by. Can be 'primary',
                        'secondary', 'tertiary', or any combination of those
                        separated by commas (eg. 'primary,secondary').
                        (default: primary,secondary)
  --pct-pix-cov PCT_PIX_COV
                        Minimum percentage of pixels to be considered a
                        candidate (default: 0.5)
  --matching-type {primary,secondary,tertiary}
                        Which color to match on for determining nearest
                        neighbors. NOTE it is always compared to the primary
                        color (eg. 'secondary' results in pairs with one
                        primary and one secondary color matching) (default:
                        primary)
  --n-matching N_MATCHING
                        Number of matches to find for each candidate. Note:
                        the same image may appear in multiple matches. Make
                        sure to give a high enough value to give room for
                        removing duplicates. (default: 3)
  --output-dir OUTPUT_DIR
                        Output directory for all output files (default:
                        output)
```
