
import warnings
import sys

from RGTools.GenomicElements import GenomicElements
from RGTools.ExogeneousSequences import ExogeneousSequences
from RGTools.BedTable import BedTable6Plus
from RGTools.SNP_utils import EnsemblRestSearch
from RGTools.utils import str2bool

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class GenomicElementExport:
    @staticmethod
    def set_parser(parser):
        subparsers = parser.add_subparsers(dest="oformat")

        parser_stat_list = subparsers.add_parser("stat_list",
                                                help="Export stat annotation values as a line-based list file.",
                                                )
        GenomicElementExport.set_parser_stat_list(parser_stat_list)

        parser_exogeneous_sequences = subparsers.add_parser("ExogeneousSequences", 
                                                          help="Export exogeneous sequences.",
                                                          )
        GenomicElementExport.set_parser_exogeneous_sequences(parser_exogeneous_sequences)

        parser_wtes = subparsers.add_parser("WTES",
                                            help="Export wild type exogeneous sequences.",
                                            )
        GenomicElementExport.set_parser_wtes(parser_wtes)

        parser_allele_expanded_es = subparsers.add_parser(
            "allele_expanded_ES",
            help="Export TRE-centered reference and allele-expanded exogeneous sequences.",
        )
        GenomicElementExport.set_parser_allele_expanded_es(parser_allele_expanded_es)

        parser_count_table = subparsers.add_parser("CountTable", 
                                                  help="Export count table.",
                                                  )
        GenomicElementExport.set_parser_count_table(parser_count_table)

        parser_heatmap = subparsers.add_parser("Heatmap", 
                                                  help="Export heatmap.",
                                                  )
        GenomicElementExport.set_parser_heatmap(parser_heatmap)

        parser_chrom_filtered_ge = subparsers.add_parser("ChromFilteredGE",
                                                         help="Export GenomicElements with only chromosome given.", 
                                                         )
        GenomicElementExport.set_parser_chrom_filtered_ge(parser_chrom_filtered_ge)

        parser_masked_ge = subparsers.add_parser("MaskedGE",
                                                 help="Export masked GenomicElements and optional masked annotations.",
                                                 )
        GenomicElementExport.set_parser_masked_ge(parser_masked_ge)

        parser_trebed = subparsers.add_parser("TREbed",
                                             help="Export TREbed file with fwdTSS and revTSS annotations.",
                                             )
        GenomicElementExport.set_parser_trebed(parser_trebed)

        parser_merged_ge = subparsers.add_parser("MergedGE",
                                                help="Merge two Genomic Element datasets.",
                                                )
        GenomicElementExport.set_parser_merged_ge(parser_merged_ge)

        parser_bed6poly = subparsers.add_parser("bed6poly",
                                                help="Export bed6 with polymorphism column from rsid names.",
                                                )
        GenomicElementExport.set_parser_bed6poly(parser_bed6poly)

    @staticmethod
    def set_parser_exogeneous_sequences(parser):
        GenomicElements.set_parser_genome(parser)
        GenomicElements.set_parser_genomic_element_region(parser)
        parser.add_argument("--opath", 
                            help="Fasta output file path.",
                            required=True,
                            )
        return parser

    @staticmethod
    def set_parser_wtes(parser):
        GenomicElements.set_parser_genome(parser)
        GenomicElements.set_parser_genomic_element_region(parser)
        parser.add_argument("--num_replicates",
                            help="Number of replicates for each region sequence.",
                            required=True,
                            type=int,
                            )
        parser.add_argument("--opath",
                            help="Output path of the WTES fasta file.",
                            required=True,
                            )
        return parser

    @staticmethod
    def set_parser_allele_expanded_es(parser):
        GenomicElements.set_parser_genome(parser)
        GenomicElements.set_parser_genomic_element_region(parser)
        parser.add_argument("--inpath_polymorphisms",
                            help="Input bed6+ polymorphism file with a bases column (e.g., REF/ALT1/ALT2).",
                            required=True,
                            type=str,
                            )
        parser.add_argument("--job_name",
                            help="Optional job name for record keeping.",
                            required=False,
                            type=str,
                            default=None,
                            )
        parser.add_argument("--opath",
                            help="Output path of the allele-expanded fasta file.",
                            required=True,
                            )
        return parser

    @staticmethod
    def set_parser_count_table(parser):
        GenomicElements.set_parser_genomic_element_region(parser)
        parser.add_argument("--opath", 
                            help="Output path of the count table.",
                            required=True,
                            )

        parser.add_argument("--sample_name", 
                            help="Sample name.",
                            action="append",
                            required=True,
                            )
        
        parser.add_argument("--stat_npy", 
                            help="Path to the stat npy file.",
                            action="append",
                            required=True,
                            )
        
        parser.add_argument("--region_id_type", 
                            help="Type of the region id (default, gene_symbol).",
                            default="default",
                            type=str,
                            choices=["default", "gene_symbol"],
                            )
        
        return parser
    
    @staticmethod
    def set_parser_heatmap(parser):
        GenomicElements.set_parser_genomic_element_region(parser)
        parser.add_argument("--track_npy", 
                            help="Path to the track npy file.",
                            action="append",
                            required=True,
                            )
        
        parser.add_argument("--title", 
                            help="Title of the heatmap.",
                            type=str,
                            action="append",
                            required=True,
                            )

        parser.add_argument("--negative", 
                            help="Whether the track is negative.",
                            type=str2bool,
                            action="append",
                            required=True,
                            )

        parser.add_argument("--per_track_max_percentile", 
                            help="Percentile used to determine the maximum value per track.",
                            type=int,
                            default=99,
                            )

        parser.add_argument("--vmax_percentile", 
                            help="Percentile used to determine the vmax.",
                            type=int,
                            default=50,
                            )

        parser.add_argument("--opath", 
                            help="Output path of the heatmap.",
                            required=True,
                            )
        return parser

    @staticmethod
    def set_parser_chrom_filtered_ge(parser):
        GenomicElements.set_parser_genomic_element_region(parser)
        parser.add_argument("--chrom_size", 
                            help="Chromosome size file.",
                            required=True,
                            )
        parser.add_argument("--opath", 
                            help="Output path of the filtered GenomicElements.",
                            required=True,
                            )
        return parser

    @staticmethod
    def set_parser_masked_ge(parser):
        GenomicElements.set_parser_genomic_element_region(parser)
        parser.add_argument("--mask_npy",
                            help="Path to boolean mask annotation npy/npz file.",
                            required=True,
                            )
        parser.add_argument("--opath",
                            help="Output path of the filtered GenomicElements.",
                            required=True,
                            )
        parser.add_argument("--anno_name",
                            help="Annotation name to export after masking. Can be specified multiple times.",
                            action="append",
                            default=[],
                            )
        parser.add_argument("--anno_npy",
                            help="Path to annotation npy/npz file. Can be specified multiple times.",
                            action="append",
                            default=[],
                            )
        parser.add_argument("--anno_type",
                            help="Annotation type for each annotation. Can be specified multiple times.",
                            action="append",
                            default=[],
                            choices=["track", "stat", "mask", "array"],
                            )
        parser.add_argument("--anno_oheader",
                            help="Output header for masked annotation files.",
                            type=str,
                            default=None,
                            )
        return parser

    @staticmethod
    def set_parser_trebed(parser):
        GenomicElements.set_parser_genomic_element_region(parser)
        parser.add_argument("--pl_sig_track", 
                            help="Path to plus strand GROcap/PROcap signal track npy/npz file.",
                            required=True,
                            )
        parser.add_argument("--mn_sig_track", 
                            help="Path to minus strand GROcap/PROcap signal track npy/npz file.",
                            required=True,
                            )
        parser.add_argument("--opath", 
                            help="Output path for the TREbed file.",
                            required=True,
                            )
        return parser

    @staticmethod
    def set_parser_merged_ge(parser):
        parser.add_argument("--left_region_file_path",
                            help="Path to the left region file.",
                            required=True,
                            type=str,
                            )
        parser.add_argument("--right_region_file_path",
                            help="Path to the right region file.",
                            required=True,
                            type=str,
                            )
        parser.add_argument("--region_file_type",
                            help="Type of the region file. "
                                 "Valid types: {}".format(
                                     list(GenomicElements.get_region_file_suffix2class_dict().keys())
                                     ),
                            required=True,
                            default="bed3",
                            type=str,
                            choices=GenomicElements.get_region_file_suffix2class_dict().keys(),
                            )
        parser.add_argument("--anno_name",
                            help="Annotation name to merge. Can be specified multiple times.",
                            action="append",
                            required=True,
                            )
        parser.add_argument("--left_anno_path",
                            help="Path to left annotation npy/npz file. Can be specified multiple times.",
                            action="append",
                            required=True,
                            )
        parser.add_argument("--right_anno_path",
                            help="Path to right annotation npy/npz file. Can be specified multiple times.",
                            action="append",
                            required=True,
                            )
        parser.add_argument("--anno_type",
                            help="Annotation type for each annotation. Can be specified multiple times.",
                            action="append",
                            required=True,
                            choices=["track", "stat", "mask", "array"],
                            )
        parser.add_argument("--oheader",
                            help="Output header for merged files.",
                            required=True,
                            type=str,
                            )
        return parser

    @staticmethod
    def set_parser_bed6poly(parser):
        parser.add_argument("--region_file_path",
                            help="Path to the input region file.",
                            required=True,
                            type=str,
                            )
        parser.add_argument("--region_file_type",
                            help="Type of the region file.",
                            required=True,
                            default="bed6",
                            type=str,
                            choices=["bed6"],
                            )
        parser.add_argument("--genome_version",
                            help="Genome version for Ensembl REST API lookup.",
                            type=str,
                            default="hg38",
                            choices=["hg38", "GRCh38", "hg19", "GRCh37"],
                            )
        parser.add_argument("--rsid_not_found_handling",
                            help="Handling strategy when an rsid is not found in Ensembl REST API.",
                            type=str,
                            default="raise",
                            choices=["raise", "drop"],
                            )
        parser.add_argument("--opath",
                            help="Output path for the bed6poly file.",
                            required=True,
                            )
        return parser

    @staticmethod
    def set_parser_stat_list(parser):
        GenomicElements.set_parser_genomic_element_region(parser)
        parser.add_argument("--stat_npy",
                            help="Path to the stat npy/npz file.",
                            required=True,
                            )
        parser.add_argument("--opath",
                            help='Output path of the list file. Use "-" or "stdout" to write to stdout.',
                            required=True,
                            )
        parser.add_argument("--dtype",
                            help="Dtype used to cast values before writing.",
                            default="str",
                            type=str,
                            choices=["str", "np.int32", "np.int64", "np.float32", "np.float64"],
                            )
        return parser

    @staticmethod
    def get_oformat_options():
        return ["stat_list", "ExogeneousSequences", "WTES", "allele_expanded_ES", "CountTable", "Heatmap", "ChromFilteredGE", "MaskedGE", "TREbed", "MergedGE", "bed6poly"]

    @staticmethod
    def export_stat_list(args):
        ge = GenomicElements(args.region_file_path,
                             args.region_file_type,
                             None,
                             )
        ge.load_region_anno_from_npy("__stat_list__", args.stat_npy, anno_type="stat")
        stat_arr = ge.get_stat_arr("__stat_list__").reshape(-1,)
        stat_arr = np.asarray(stat_arr, dtype=eval(args.dtype))

        output_handle = sys.stdout if args.opath in {"-", "stdout"} else open(args.opath, "w")
        try:
            for value in stat_arr:
                output_handle.write(f"{value}\n")
        finally:
            if output_handle is not sys.stdout:
                output_handle.close()

    @staticmethod
    def export_exogeneous_sequences(args):
        ge = GenomicElements(args.region_file_path, 
                             args.region_file_type, 
                             args.fasta_path, 
                             )
        ge.export_exogeneous_sequences(args.opath)

    @staticmethod
    def export_wtes(args):
        if args.num_replicates < 1:
            raise ValueError(f"num_replicates must be >= 1, got {args.num_replicates}")

        ge = GenomicElements(args.region_file_path,
                             args.region_file_type,
                             args.fasta_path,
                             )
        regions = list(ge.get_region_bed_table().iter_regions())
        region_seqs = ge.get_all_region_seqs()

        seq_ids = []
        seqs = []
        for region, seq in zip(regions, region_seqs):
            for ind in range(args.num_replicates):
                seq_ids.append(f"{region['chrom']}:{region['start']}-{region['end']}_{ind}")
                seqs.append(seq)

        ExogeneousSequences.write_sequences_to_fasta(seq_ids, seqs, args.opath)

    @staticmethod
    def export_allele_expanded_es(args):
        ge = GenomicElements(args.region_file_path,
                             args.region_file_type,
                             args.fasta_path,
                             )

        polymorphisms_bt = BedTable6Plus(extra_column_names=["bases"],
                                         extra_column_dtype=[str],
                                         enable_sort=False,
                                         )
        polymorphisms_bt.load_from_file(args.inpath_polymorphisms)

        output_seq_ids = []
        output_sequences = []
        for region in ge.get_region_bed_table().iter_regions():
            overlapping_snp_bt = polymorphisms_bt.region_subset(region["chrom"],
                                                                region["start"],
                                                                region["end"],
                                                                )
            if len(overlapping_snp_bt) == 0:
                continue

            ref_sequence = ge.get_region_seq(region["chrom"], region["start"], region["end"])
            if ref_sequence is None:
                raise ValueError(
                    f"Chromosome {region['chrom']} not found in genome file. "
                    f"Cannot export region {region['chrom']}:{region['start']}-{region['end']}"
                )

            output_seq_ids.append(f"{region['chrom']}_{region['start']}_{region['end']}_ref")
            output_sequences.append(ref_sequence)

            for snp in overlapping_snp_bt.iter_regions():
                index2mut = snp["start"] - region["start"]
                if index2mut < 0 or index2mut >= len(ref_sequence):
                    continue
                ref_base = ref_sequence[index2mut].upper()

                for base in str(snp["bases"]).split("/"):
                    mutated_base = base.upper().strip()
                    if len(mutated_base) != 1:
                        continue
                    if mutated_base == ref_base:
                        continue

                    mutated_seq = ref_sequence[:index2mut] + mutated_base + ref_sequence[index2mut+1:]
                    output_seq_ids.append(
                        f"{region['chrom']}_{region['start']}_{region['end']}_{snp['start']}:{ref_base}2{mutated_base}"
                    )
                    output_sequences.append(mutated_seq)

        ExogeneousSequences.write_sequences_to_fasta(output_seq_ids, output_sequences, args.opath)

    @staticmethod
    def region2region_id(region, region_id_type):
        '''
        Return the region id.

        Keyword arguments:
        - region: Region object.
        - region_id_type: Type of the region id (default, gene_symbol).

        Returns:
        - region_id: Region id.
        '''
        if region_id_type == "default":
            return region["chrom"] + ":" + str(region["start"]) + "-" + str(region["end"])
        elif region_id_type == "gene_symbol":
            if "gene_symbol" not in region.get_fields():
                raise ValueError(f"gene_symbol not supported for this region file type.")
            return region["gene_symbol"]
        else:
            raise ValueError(f"Invalid region id type: {region_id_type}")

    @staticmethod
    def export_count_table(args):
        if len(args.sample_name) != len(args.stat_npy):
            raise ValueError(f"Number of sample names ({len(args.sample_name)}) must match number of stat_npy files ({len(args.stat_npy)})")
        
        ge = GenomicElements(args.region_file_path, 
                             args.region_file_type, 
                             None, 
                             )

        for sample_name, stat_npy in zip(args.sample_name, args.stat_npy):
            ge.load_region_anno_from_npy(sample_name, stat_npy, anno_type="stat")
        
        region_names = []
        for region in ge.get_region_bed_table().iter_regions():
            region_names.append(GenomicElementExport.region2region_id(region, args.region_id_type))

        output_df = pd.DataFrame(columns=args.sample_name, 
                                 index=region_names,
                                 )

        for sample_name in args.sample_name:
            output_df[sample_name] = ge.get_stat_arr(sample_name).reshape(-1,)

        output_df.to_csv(args.opath, 
                         index=True,
                         )

    @staticmethod
    def get_heatmap_vmin_vmax(track_arr, per_track_max_percentile, 
                              vmax_percentile):
        '''
        Get the vmin and vmax for the heatmap.
        This function first determines the maximum value per track, 
        then determines the vmax based on percentile of the determined 
        maximum values.

        Keyword arguments:
        - track_arr: Track array. Must be positive.
        - per_track_max_percentile: Percentile used to determine the maximum value per track.
        - vmax_percentile: Percentile used to determine the vmax.

        Returns:
        - vmin: Vmin.
        - vmax: Vmax.
        '''
        vmin=0
        top_1p_signal_per_track = np.percentile(track_arr, per_track_max_percentile, axis=1)
        vmax = np.percentile(top_1p_signal_per_track, vmax_percentile)

        if vmax==0: 
            vmax = 1

        return vmin, vmax

    @staticmethod
    def track_list_to_arr(track_list):
        max_len = max(len(track) for track in track_list) if len(track_list) > 0 else 0
        track_arr = np.zeros((len(track_list), max_len))
        for i, track in enumerate(track_list):
            track_arr[i, :len(track)] = track
        return track_arr

    @staticmethod
    def plot_heatmap_image(ax, track_arr, sort_idx, plot_cmap, 
                           title, per_track_max_percentile, 
                           vmax_percentile):
        '''
        Plot a heatmap with imshow.

        Keyword arguments:
        - ax: Axes object.
        - track_arr: Track array. Must be positive.
        - sort_idx: Sort index.
        - plot_cmap: Plot cmap.
        - title: Title.
        - per_track_max_percentile: Percentile used to determine the maximum value per track.
        - vmax_percentile: Percentile used to determine the vmax.

        Returns:
        - imshow_pos: Imshow position.
        '''
        vmin, vmax = GenomicElementExport.get_heatmap_vmin_vmax(track_arr, 
                                                                per_track_max_percentile, 
                                                                vmax_percentile, 
                                                                )

        imshow_pos = ax.imshow(track_arr[sort_idx, :], 
                               cmap=plot_cmap,
                               aspect="auto",
                               vmin=vmin,
                               vmax=vmax,
                               )
        
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(title)

        return imshow_pos

    @staticmethod
    def plot_heatmap_mean(ax, track_arr, negative_sig):
        '''
        Plot the mean signal.
        '''
        width = track_arr.shape[1]
        ax.plot(np.arange(width), 
                - track_arr.mean(axis=0) if negative_sig else track_arr.mean(axis=0),
                color="black",
                )
        ax.set_ylabel("Mean signal")
        ax.set_xlabel("Position")
        ax.set_xticks([0, width/2, width])
        ax.set_xticklabels([-width//2, 0, width//2])

    @staticmethod
    def export_heatmap(args):
        if len(args.title) != len(args.track_npy):
            raise ValueError(f"Number of titles ({len(args.title)}) must match number of track_npy files ({len(args.track_npy)})")
        if len(args.title) != len(args.negative):
            raise ValueError(f"Number of titles ({len(args.title)}) must match number of negative flags ({len(args.negative)})")
        
        ge = GenomicElements(args.region_file_path, 
                             args.region_file_type, 
                             None, 
                             )
        for track_title, track_npy in zip(args.title, args.track_npy):
            ge.load_region_anno_from_npy(track_title, track_npy, anno_type="track")

        track_arr_list = [
            np.abs(GenomicElementExport.track_list_to_arr(ge.get_track_list(track_title)))
            for track_title in args.title
        ]

        fig, ax = plt.subplots(2, len(args.title), 
                               figsize=(4 * len(args.title), 8),
                               height_ratios=[1, 0.2],
                               squeeze=False,
                               )

        # Figure out sorting
        sort_idx = np.argsort(np.concatenate(track_arr_list, axis=1).max(axis=1))

        for ind, track_title in enumerate(args.title):
            track_arr = np.abs(GenomicElementExport.track_list_to_arr(ge.get_track_list(track_title)))

            if args.negative[ind]:
                plot_cmap = "Blues"
            else:
                plot_cmap = "Reds"

            imshow_pos = GenomicElementExport.plot_heatmap_image(ax[0, ind], 
                                                                 track_arr, 
                                                                 sort_idx, 
                                                                 plot_cmap, 
                                                                 track_title,
                                                                 args.per_track_max_percentile, 
                                                                 args.vmax_percentile,
                                                                 )

            GenomicElementExport.plot_heatmap_mean(ax[1, ind], 
                                                   track_arr, 
                                                   args.negative[ind],
                                                   )

        fig.tight_layout()
        fig.savefig(args.opath)

    @staticmethod
    def export_chrom_filtered_ge(args):
        ge = GenomicElements(args.region_file_path, 
                             args.region_file_type, 
                             None, 
                             )
        chrom_size_df = pd.read_csv(args.chrom_size, 
                                    sep="\t", 
                                    header=None, 
                                    names=["chrom", "size"], 
                                    )
        chrom_size_dict = dict(zip(chrom_size_df["chrom"], chrom_size_df["size"]))

        filter_logical = [c in chrom_size_dict.keys() for c in ge.get_region_bed_table().get_chrom_names()]
        output_bt = ge.get_region_bed_table().apply_logical_filter(filter_logical)
        output_bt.write(args.opath)

    @staticmethod
    def export_masked_ge(args):
        if len(args.anno_name) != len(args.anno_npy):
            raise ValueError(
                f"Number of anno_name ({len(args.anno_name)}) must match "
                f"number of anno_npy ({len(args.anno_npy)})"
            )
        if len(args.anno_name) != len(args.anno_type):
            raise ValueError(
                f"Number of anno_name ({len(args.anno_name)}) must match "
                f"number of anno_type ({len(args.anno_type)})"
            )
        if len(args.anno_name) > 0 and args.anno_oheader is None:
            raise ValueError("anno_oheader is required when annotation export arguments are provided.")

        ge = GenomicElements(args.region_file_path,
                             args.region_file_type,
                             None,
                             )
        ge.load_region_anno_from_npy("__mask__", args.mask_npy, anno_type="mask")
        mask_arr = ge.get_mask_arr("__mask__").reshape(-1,)

        for anno_name, anno_npy, anno_type in zip(args.anno_name, args.anno_npy, args.anno_type):
            ge.load_region_anno_from_npy(anno_name, anno_npy, anno_type=anno_type)

        result_ge = ge.apply_logical_filter(mask_arr, args.opath)

        for anno_name in args.anno_name:
            result_ge.save_anno_npy(anno_name, f"{args.anno_oheader}.{anno_name}.npy")

    @staticmethod
    def export_trebed(args):
        '''
        Export TREbed file with forward and reverse TSS annotations.
        
        Examines GROcap/PROcap signal tracks to find TSS positions:
        - fwdTSS: Absolute genomic position of maximum signal in plus strand track
        - revTSS: Absolute genomic position of maximum signal in minus strand track
        '''
        ge = GenomicElements(args.region_file_path, 
                             args.region_file_type, 
                             None, 
                             )
        
        ge.load_region_anno_from_npy("pl_track", args.pl_sig_track, anno_type="track")
        ge.load_region_anno_from_npy("mn_track", args.mn_sig_track, anno_type="track")
        
        pl_track_list = ge.get_track_list("pl_track")
        mn_track_list = ge.get_track_list("mn_track")
        
        # Create TREbed output
        trebed_bt = GenomicElements.BedTableTREBed(enable_sort=False)
        
        output_dict_list = []
        for i, region in enumerate(ge.get_region_bed_table().iter_regions()):
            pl_track = np.abs(pl_track_list[i])
            mn_track = np.abs(mn_track_list[i])
            
            # Find TSS positions (position with maximum signal, relative to region start)
            # fwdTSS: forward TSS from plus strand track
            # revTSS: reverse TSS from minus strand track
            fwdTSS_rel_pos = np.argmax(pl_track) if len(pl_track) > 0 else 0
            revTSS_rel_pos = np.argmax(mn_track) if len(mn_track) > 0 else 0
            
            # Convert to absolute genomic coordinates
            fwdTSS = region["start"] + fwdTSS_rel_pos
            revTSS = region["start"] + revTSS_rel_pos
            
            # Create region name (default: chrom:start-end)
            region_name = f"{region['chrom']}:{region['start']}-{region['end']}"
            
            output_dict = {
                "chrom": region["chrom"],
                "start": region["start"],
                "end": region["end"],
                "name": region_name,
                "fwdTSS": fwdTSS,
                "revTSS": revTSS,
            }
            output_dict_list.append(output_dict)
        
        trebed_bt.load_from_dataframe(pd.DataFrame(output_dict_list))
        trebed_bt.write(args.opath)

    @staticmethod
    def export_merged_ge(args):
        if len(args.anno_name) != len(args.left_anno_path):
            raise ValueError(
                f"Number of anno_name ({len(args.anno_name)}) must match "
                f"number of left_anno_path ({len(args.left_anno_path)})"
            )
        if len(args.anno_name) != len(args.right_anno_path):
            raise ValueError(
                f"Number of anno_name ({len(args.anno_name)}) must match "
                f"number of right_anno_path ({len(args.right_anno_path)})"
            )
        if len(args.anno_name) != len(args.anno_type):
            raise ValueError(
                f"Number of anno_name ({len(args.anno_name)}) must match "
                f"number of anno_type ({len(args.anno_type)})"
            )

        left_ge = GenomicElements(args.left_region_file_path,
                                  args.region_file_type,
                                  None,
                                  )
        right_ge = GenomicElements(args.right_region_file_path,
                                   args.region_file_type,
                                   None,
                                   )

        for anno_name, anno_type, left_anno_path, right_anno_path in zip(args.anno_name,
                                                                          args.anno_type,
                                                                          args.left_anno_path,
                                                                          args.right_anno_path):
            left_ge.load_region_anno_from_npy(anno_name, left_anno_path, anno_type=anno_type)
            right_ge.load_region_anno_from_npy(anno_name, right_anno_path, anno_type=anno_type)

        output_region_path = args.oheader + "." + args.region_file_type
        merged_ge = GenomicElements.merge_genomic_elements(left_ge,
                                                           right_ge,
                                                           output_region_path,
                                                           args.anno_name,
                                                           sort_new_ge=True,
                                                           )

        for anno_name in args.anno_name:
            merged_ge.save_anno_npy(anno_name, args.oheader + "." + anno_name + ".npy")

    @staticmethod
    def export_bed6poly(args):
        ge = GenomicElements(args.region_file_path,
                             args.region_file_type,
                             None,
                             )
        search_engine = EnsemblRestSearch(genome_version=args.genome_version, species="human")

        output_dict_list = []
        for region in ge.get_region_bed_table().iter_regions():
            rsid = str(region["name"])

            try: 
                snp_info = search_engine.get_rsid_snp_simple_info(rsid)
            except Exception as e:
                if args.rsid_not_found_handling == "drop":
                    warnings.warn(f"Dropping region for rsid not found: {rsid}")
                    continue
                raise Exception(f"Error getting SNP info for rsid: {rsid}") from e

            polymorphism = snp_info["bases"]

            if snp_info["chrom"] != region["chrom"] or snp_info["start"] != region["start"] or snp_info["end"] != region["end"]:
                warnings.warn(f"Position mismatch for rsid: {rsid}.\n"
                    f"SNP info: {snp_info['chrom']}:{snp_info['start']}-{snp_info['end']}.\n"
                    f"Region: {region['chrom']}:{region['start']}-{region['end']}.\n"
                )

            output_dict = {
                "chrom": region["chrom"],
                "start": region["start"],
                "end": region["end"],
                "name": region["name"],
                "score": region["score"],
                "strand": region["strand"],
                "polymorphism": polymorphism,
            }
            output_dict_list.append(output_dict)

        output_bt = BedTable6Plus(extra_column_names=["polymorphism"],
                                  extra_column_dtype=[str],
                                  enable_sort=False,
                                  )
        output_bt.load_from_dataframe(pd.DataFrame(output_dict_list))
        output_bt.write(args.opath)

    @staticmethod
    def main(args):
        if args.oformat == "stat_list":
            GenomicElementExport.export_stat_list(args)
        elif args.oformat == "ExogeneousSequences":
            GenomicElementExport.export_exogeneous_sequences(args)
        elif args.oformat == "WTES":
            GenomicElementExport.export_wtes(args)
        elif args.oformat == "allele_expanded_ES":
            GenomicElementExport.export_allele_expanded_es(args)
        elif args.oformat == "CountTable":
            GenomicElementExport.export_count_table(args)
        elif args.oformat == "Heatmap":
            GenomicElementExport.export_heatmap(args)
        elif args.oformat == "ChromFilteredGE":
            GenomicElementExport.export_chrom_filtered_ge(args)
        elif args.oformat == "MaskedGE":
            GenomicElementExport.export_masked_ge(args)
        elif args.oformat == "TREbed":
            GenomicElementExport.export_trebed(args)
        elif args.oformat == "MergedGE":
            GenomicElementExport.export_merged_ge(args)
        elif args.oformat == "bed6poly":
            GenomicElementExport.export_bed6poly(args)
        else:
            raise ValueError(f"Invalid output format: {args.oformat}")
