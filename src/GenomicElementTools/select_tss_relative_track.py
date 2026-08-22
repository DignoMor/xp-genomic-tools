"""Select a TSS-relative track score and emit coordinate/mask annotations."""

from __future__ import annotations

import math

import numpy as np

from RGTools.GenomicElements import GenomicElements
from RGTools.TSSRelativeCoordinates import (
    iter_relaxed_window,
    tss_relative_to_track_index,
)


class SelectTssRelativeTrack:
    @staticmethod
    def set_parser(parser):
        GenomicElements.set_parser_genomic_element_region(parser)

        parser.add_argument(
            "--track_npy",
            help="Path to aligned numeric track array (.npy or single-array .npz).",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--strand",
            help="Selected TSS strand: '+' uses fwdTSS; '-' uses revTSS.",
            required=True,
            type=str,
            choices=["+", "-"],
        )
        parser.add_argument(
            "--target_coord",
            help="Nonzero TSS-relative coordinate to evaluate.",
            required=True,
            type=int,
        )
        parser.add_argument(
            "--relaxation",
            help="Nonnegative relaxation radius around target_coord (default 0).",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--min_score",
            help="Inclusive finite minimum score for a match.",
            required=True,
            type=float,
        )
        parser.add_argument(
            "--track_window_size",
            help="Scored window width in bases (default 1 for point tracks).",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--coordinate_opath",
            help="Output path for selected TSS-relative coordinates (.npy).",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--mask_opath",
            help="Output path for match mask (.npy).",
            required=True,
            type=str,
        )

    @staticmethod
    def main(args):
        if args.region_file_type != "TREbed":
            raise ValueError(
                f"select_tss_relative_track requires region_file_type 'TREbed'; "
                f"got {args.region_file_type!r}."
            )
        if args.strand not in ("+", "-"):
            raise ValueError(
                f"strand must be '+' or '-', found {args.strand!r}."
            )
        if args.relaxation < 0:
            raise ValueError(
                f"relaxation must be a nonnegative integer, found {args.relaxation}."
            )
        if args.track_window_size < 1:
            raise ValueError(
                "track_window_size must be a positive integer, found "
                f"{args.track_window_size}."
            )
        if args.target_coord == 0:
            raise ValueError("TSS-relative target_coord zero is invalid.")
        if not math.isfinite(args.min_score):
            raise ValueError(
                f"min_score must be finite; got {args.min_score!r}."
            )

        ge = GenomicElements(
            region_file_path=args.region_file_path,
            region_file_type=args.region_file_type,
            fasta_path=None,
        )
        ge.load_region_anno_from_npy("track", args.track_npy, anno_type="track")

        bed = ge.get_region_bed_table()
        n = ge.get_num_regions()
        if n > 0:
            sample_track = ge.get_region_track_by_index("track", 0)
            if sample_track.dtype == np.bool_ or np.issubdtype(sample_track.dtype, np.bool_):
                raise ValueError("Boolean tracks are rejected; provide a numeric track.")
            if not np.issubdtype(sample_track.dtype, np.number):
                raise ValueError(
                    f"Track must be numeric; got dtype {sample_track.dtype}."
                )

        coords = np.zeros((n, 1), dtype=np.int64)
        mask = np.zeros((n, 1), dtype=bool)
        tss_field = "fwdTSS" if args.strand == "+" else "revTSS"
        window_coords = list(iter_relaxed_window(args.target_coord, args.relaxation))

        for i, region in enumerate(bed.iter_regions()):
            start = int(region["start"])
            end = int(region["end"])
            selected_tss = int(region[tss_field])

            if selected_tss == -1:
                continue

            if not (start <= selected_tss < end):
                raise ValueError(
                    f"Selected {tss_field}={selected_tss} is outside interval "
                    f"[{start}, {end}) for row {i}."
                )

            track_row = ge.get_region_track_by_index("track", i)
            best_coord = None
            best_score = None
            for coord in window_coords:
                index = tss_relative_to_track_index(
                    strand=args.strand,
                    coord=coord,
                    start=start,
                    end=end,
                    tss=selected_tss,
                    track_window_size=args.track_window_size,
                )
                score_f = float(track_row[index])
                if math.isnan(score_f):
                    raise ValueError(
                        f"NaN score at row {i}, track index {index}."
                    )
                if best_score is None or score_f > best_score:
                    best_score = score_f
                    best_coord = coord

            if best_score is not None and best_score >= args.min_score:
                coords[i, 0] = int(best_coord)
                mask[i, 0] = True

        ge.load_region_stat_from_arr("tss_rel_coord", coords)
        ge.load_mask_from_arr("tss_rel_mask", mask)
        ge.save_anno_npy("tss_rel_coord", args.coordinate_opath)
        ge.save_anno_npy("tss_rel_mask", args.mask_opath)
