import json
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from convert import shift_mountain
from histograms import LuvHistogram, Uv
from image_util import image_histo, resize
from mountain import Mountain, default_params


class LoadedImage(NamedTuple):
    fpath: Path
    image: Image.Image
    histo: LuvHistogram
    smoothed: LuvHistogram
    mountains: List[Mountain]


def images(
    dirs: List[Path],
    max_size: int | None = None,
    sigma=1,
    white_threshold=10,
    mountain_params=default_params,
):
    """Images performs the color cluster detection and peak color identification"""

    def load(f: Path):
        image = resize(f, max_size)
        histo = image_histo(image).without_white(white_threshold)
        smoothed = histo.collapse_L().smooth(sigma)
        mountains = smoothed.mountains(
            mountain_params,
            4,  # white, primary, secondary, tertiary
        )

        return LoadedImage(f, image, histo, smoothed, mountains)

    images: List[Path] = []
    for d in dirs:
        for ext in ["png", "jpg", "jpeg"]:
            for f in d.glob(f"*.{ext}"):
                images.append(f)

    print(f"Loading {len(images)} images")

    with ThreadPoolExecutor() as pool:
        processing: List[Future[LoadedImage]] = []
        for f in images:
            processing.append(pool.submit(load, f))

        df = []

        def skip(loaded: LoadedImage, msg: str):
            print(f"Skipping {loaded.fpath}: {msg}")
            df.append(
                [
                    loaded.fpath,
                    loaded.histo.total(),
                    0,
                    "",
                    *(None, None, None),
                    0,
                    "",
                    *(None, None, None),
                    0,
                    msg,
                ]
            )

        print(f"Processing {len(processing)} images")
        for complete in tqdm(as_completed(processing), total=len(processing)):
            try:
                loaded = complete.result()
            except Exception as e:
                print(f"Failed to load {complete}: {e}")
                continue

            if len(loaded.mountains) == 0:
                skip(loaded, f"skipping due to no mountains")
                continue

            mountains = list(loaded.mountains)

            def next_cluster():
                if len(mountains) == 0:
                    return (
                        Mountain((0, 0, 0), (0, 0, 0)),
                        (None, None, None),
                        "#FFFFFF",
                    )

                mtn = mountains.pop(0)
                if (0, 0) in mtn:
                    return next_cluster()

                c = loaded.histo.max_L(Uv(*mtn.peak))
                return (mtn, c, c.hex())

            try:
                p_mtn, primary, primary_hex = next_cluster()
            except ValueError as e:
                skip(loaded, f"failed to find primary: {e}")
                continue

            if primary[0] is None:
                skip(loaded, f"skipping due to only one mountain (excluded, white)")
                continue

            s_mtn, secondary, secondary_hex = next_cluster()
            t_mtn, tertiary, tertiary_hex = next_cluster()

            dist = (
                np.linalg.norm(np.asarray(primary) - np.asarray(secondary))
                if secondary[0] is not None
                else 0
            )
            uv = (
                loaded.histo.collapse_L()
            )  # use mountain size on non-smoothed histogram to get the true size

            df.append(
                [
                    loaded.fpath,
                    loaded.histo.total(),
                    dist,
                    primary_hex,
                    *primary,
                    uv.mountain_size(p_mtn) / loaded.histo.total(),
                    secondary_hex,
                    *secondary,
                    uv.mountain_size(s_mtn) / loaded.histo.total(),
                    tertiary_hex,
                    *tertiary,
                    uv.mountain_size(t_mtn) / loaded.histo.total(),
                    "",
                ]
            )

    return pd.DataFrame(
        df,
        columns=[
            "image",
            "pixels",
            "primary_secondary_dist",
            "primary",
            "primary_L",
            "primary_u",
            "primary_v",
            "primary_pct",
            "secondary",
            "secondary_L",
            "secondary_u",
            "secondary_v",
            "secondary_pct",
            "tertiary",
            "tertiary_L",
            "tertiary_u",
            "tertiary_v",
            "tertiary_pct",
            "msg",
        ],
    )


def candidates(
    images_df: pd.DataFrame, percent_pixel_coverage: float, filter_color: str
):
    images_df["total_pct"] = 0
    if "primary" in filter_color:
        images_df["total_pct"] += images_df["primary_pct"]
    if "secondary" in filter_color:
        images_df["total_pct"] += images_df["secondary_pct"]
    if "tertiary" in filter_color:
        images_df["total_pct"] += images_df["tertiary_pct"]

    images_df["candidate"] = images_df["total_pct"] >= percent_pixel_coverage


def neighbors(images_df: pd.DataFrame, matching_type: str, n_neighbors: int):
    primary_nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    images_df = images_df.dropna(
        subset=[f"{matching_type}_L", f"{matching_type}_u", f"{matching_type}_v"]
    )
    primary_nn.fit(
        images_df[[f"{matching_type}_L", f"{matching_type}_u", f"{matching_type}_v"]]
    )

    candidates_df = images_df[images_df["candidate"]]

    dist, idxs = primary_nn.kneighbors(
        candidates_df[["primary_L", "primary_u", "primary_v"]],
    )

    df = []
    for i, (d, idx) in enumerate(zip(dist, idxs)):
        cand_name = candidates_df.iloc[i]["image"]
        row = [
            cand_name,
        ]
        for n_idx, n_d in zip(idx, d):
            if len(row) >= 2 * n_neighbors + 1:
                break

            # Don't match to self, especially if matching_type is 'primary'
            img_name = images_df.iloc[n_idx]["image"]
            if img_name == cand_name:
                continue

            row.extend([images_df.iloc[n_idx]["image"], n_d])

        while len(row) < 2 * n_neighbors + 1:
            row.extend(["", np.nan])

        df.append(row)

    columns = ["target"]
    for i in range(1, n_neighbors + 1):
        columns.extend([f"{matching_type}{i}", f"{matching_type}{i}_dist"])

    return pd.DataFrame(
        df,
        columns=columns,
    )


def transform_all(
    images_df: pd.DataFrame,
    neighbors_df: pd.DataFrame,
    output_dir: Path,
    matching_type: str,
    max_size: int,
    sigma=1,
    white_threshold=10,
):
    neighbor_columns = [
        col
        for col in neighbors_df.columns
        if col.startswith(matching_type) and not col.endswith("dist")
    ]

    def transform(neighbors: pd.Series):
        target = Path(neighbors["target"])
        root = (output_dir / neighbors["target"]).with_suffix("")
        root.mkdir(exist_ok=True, parents=True)

        try:
            img = resize(target, max_size)
            img.save(root / "target.png")

            target_row = images_df[images_df["image"] == str(target)].iloc[0]

            extra_data = {}

            target_uv = Uv(
                target_row[f"{matching_type}_u"], target_row[f"{matching_type}_v"]
            )

            try:
                for col in tqdm(
                    neighbor_columns, desc=f"Transforming {target}", leave=False
                ):
                    if neighbors[col] == "":
                        continue
                    try:
                        neighbor = Path(neighbors[col])
                        n_img = resize(neighbor, max_size)

                        neighbor_row = images_df[
                            images_df["image"] == str(neighbor)
                        ].iloc[0]

                        neighbor_histo = image_histo(n_img).without_white(
                            white_threshold
                        )
                        n_histo_uv = neighbor_histo.collapse_L().smooth(sigma)
                        mtn = n_histo_uv.mountain(
                            default_params,
                            Uv(neighbor_row["primary_u"], neighbor_row["primary_v"]),
                        )
                        n_img, converted = shift_mountain(n_img, mtn, target_uv)
                        extra_data[f"{col}_extra"] = {
                            "converted_pix_count": converted,
                            "pixels": int(neighbor_row["pixels"]),
                            "primary_L": neighbor_row["primary_L"],
                            "primary_u": neighbor_row["primary_u"],
                            "primary_v": neighbor_row["primary_v"],
                        }
                        n_img.save(root / f"{col}.png")
                    except Exception as e:
                        raise Exception(
                            f"Failed to transform {col} ({neighbor})"
                        ) from e
            finally:
                data = neighbors.to_dict()
                data.update(target_row.to_dict())
                data.update(extra_data)
                if "msg" in data and np.isnan(data["msg"]):
                    del data["msg"]

                data_path = Path(root / "data.json")
                with open(data_path, "w") as f:
                    json.dump(data, f, indent=2)

            return neighbors
        except Exception as e:
            raise Exception(f"Failed to transform {target}") from e

    with ThreadPoolExecutor() as pool:
        processing: List[Future[LoadedImage]] = []
        print(f"Transforming matches for {len(neighbors_df)} images")

        # start the bar before submitting tasks so that it shows at the top
        bar = tqdm(total=len(neighbors_df))

        for i, neighbors in neighbors_df.iterrows():
            processing.append(pool.submit(transform, neighbors))

        for complete in as_completed(processing):
            bar.update()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        type=Path,
        help="Directories to search for images, if the images csv is not found",
    )
    parser.add_argument(
        "--img-max-size",
        type=int,
        default=160,
        help="Resize images to this standard size. For non-square images, the maximum dimension is used.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("images.csv"),
        help="Path to save/load images and their color clusters. This takes long to compute so it is recommended to reuse one.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help=(
            "Path to save/load candidates csv."
            "If you wish to apply filters on the candidates before generating the matching images, "
            "pair this with '--candidate-only' first to save this file and stop execution. "
            "After making changes, run it again without '--candidate-only' and it will use the modified candidates."
        ),
    )
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Only generate the candidates csv and stop execution",
    )
    parser.add_argument(
        "--filter-color",
        type=str,
        default="primary,secondary",
        help="Which color to filter candidates by. Can be 'primary', 'secondary', 'tertiary', or any combination of those separated by commas (eg. 'primary,secondary').",
    )
    parser.add_argument(
        "--pct-pix-cov",
        type=float,
        default=0.5,
        help="Minimum percentage of pixels to be considered a candidate",
    )
    parser.add_argument(
        "--matching-type",
        default="primary",
        choices=["primary", "secondary", "tertiary"],
        help="Which color to match on for determining nearest neighbors. NOTE it is always compared to the primary color (eg. 'secondary' results in pairs with one primary and one secondary color matching)",
    )
    parser.add_argument(
        "--n-matching",
        type=int,
        default=3,
        help="Number of matches to find for each candidate. Note: the same image may appear in multiple matches. Make sure to give a high enough value to give room for removing duplicates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for all output files",
    )

    args = parser.parse_args()

    if args.images.exists():
        images_df = pd.read_csv(args.images)
    else:
        print(
            f"Loading images from {args.dirs} (caching resized images to 'resized' and histograms to 'histograms')"
        )
        images_df = images(args.dirs, args.img_max_size)
        images_df.to_csv(args.images, index=False)
        print(f"Writing image cluster detection to {args.images}")

    print(f"Read {len(images_df)} images")

    if args.candidates is not None and args.candidates.exists():
        images_df = pd.read_csv(args.candidates)
    else:
        candidates(images_df, args.pct_pix_cov, args.filter_color)

    print(
        f"Found {images_df['candidate'].sum()} candidates (of {len(images_df)} total)"
    )

    if args.candidates is not None:
        images_df.to_csv(args.candidates, index=False)

    if args.candidate_only:
        print("Candidates generated, stopping due to --candidate-only")
        sys.exit(0)

    neighbors_df = neighbors(images_df, args.matching_type, args.n_matching)

    output = Path(args.output_dir)
    output.mkdir(exist_ok=True, parents=True)

    neighbors_df.to_csv(output / "neighbors.csv", index=False)

    transform_all(
        images_df, neighbors_df, args.output_dir, args.matching_type, args.img_max_size
    )
