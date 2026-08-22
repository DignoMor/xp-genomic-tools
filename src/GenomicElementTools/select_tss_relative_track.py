"""Select a TSS-relative track score and emit coordinate/mask annotations."""

from __future__ import annotations

import math

import numpy as np

from RGTools.GenomicElements import GenomicElements
from RGTools.TSSRelativeCoordinates import tss_relative_to_track_index


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
        if args.strand != "+":
            raise ValueError(
                f"Strand {args.strand!r} selection is not yet delivered in this "
                "release slice; only '+' is supported."
            )
        if args.relaxation < 0:
            raise ValueError(
                f"relaxation must be a nonnegative integer, found {args.relaxation}."
            )
        if args.relaxation != 0:
            raise ValueError(
                f"relaxation={args.relaxation} is not yet delivered in this "
                "release slice; only relaxation=0 is supported."
            )
        if args.track_window_size < 1:
            raise ValueError(
                "track_window_size must be a positive integer, found "
                f"{args.track_window_size}."
            )
        if args.track_window_size != 1:
            raise ValueError(
                f"track_window_size={args.track_window_size} is not yet delivered "
                "in this release slice; only track_window_size=1 is supported."
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

        for i, region in enumerate(bed.iter_regions()):
            start = int(region["start"])
            end = int(region["end"])
            fwd_tss = int(region["fwdTSS"])

            if fwd_tss == -1:
                continue

            if not (start <= fwd_tss < end):
                raise ValueError(
                    f"Selected fwdTSS={fwd_tss} is outside interval [{start}, {end}) "
                    f"for row {i}."
                )

            index = tss_relative_to_track_index(
                strand="+",
                coord=args.target_coord,
                start=start,
                end=end,
                tss=fwd_tss,
                track_window_size=1,
            )
            score = ge.get_region_track_by_index("track", i)[index]
            score_f = float(score)
            if math.isnan(score_f):
                raise ValueError(
                    f"NaN score at row {i}, track index {index}."
                )
            if score_f >= args.min_score:
                coords[i, 0] = int(args.target_coord)
                mask[i, 0] = True

        ge.load_region_stat_from_arr("tss_rel_coord", coords)
        ge.load_mask_from_arr("tss_rel_mask", mask)
        ge.save_anno_npy("tss_rel_coord", args.coordinate_opath)
        ge.save_anno_npy("tss_rel_mask", args.mask_opath)
