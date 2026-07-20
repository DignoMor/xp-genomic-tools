import argparse

import numpy as np

from RGTools.ExogeneousSequences import ExogeneousSequences
from RGTools.MemeMotif import MemeMotif
from RGTools.utils import str2bool

from RGTools.utils import reverse_complement as RC

class Motif:
    @staticmethod
    def set_parser_motif_search(parser: argparse.ArgumentParser):
        ExogeneousSequences.set_parser_exogeneous_sequences(parser)

        parser.add_argument("--motif_file", 
                            type=str,
                            required=True,
                            help="The file containing the motifs to search for.",
                            )
        
        parser.add_argument("--output_header", 
                            type=str,
                            required=True,
                            help="The header of the output file.",
                            )
        
        parser.add_argument("--estimate_background_freq",
                            help="Estimate background frequency from the sequence.",
                            type=str2bool, 
                            default=True,
                            )
        
        parser.add_argument("--reverse_complement",
                            help="Reverse complement the sequence while matching for motifs.",
                            type=str2bool,
                            default=False,
                            )

    @staticmethod
    def motif_search_main(args: argparse.Namespace):
        input_es = ExogeneousSequences(args.fasta)
        motif_dataset = MemeMotif(args.motif_file)

        seq_list = input_es.get_all_region_seqs()
        

        for motif in motif_dataset.get_motif_list():
            motif_pwm = motif_dataset.get_motif_pwm(motif)
            motif_alphabet = motif_dataset.get_alphabet()
            output_track_list = []

            # Estimate background frequency
            if not args.estimate_background_freq:
                bg_freq = motif_dataset.get_bg_freq()
                bg_freq = np.array(bg_freq, dtype=np.float64)

                # still estimate N frequency from the sequence
                full_str = "".join([s.upper() for s in seq_list])
                if "N" in full_str:
                    motif_alphabet += "N"
                    n_count = np.char.count(full_str, "N")
                    bg_freq = np.concatenate((bg_freq, 
                                              [n_count / (len(full_str) - n_count)], 
                                              ))
                    bg_freq /= np.sum(bg_freq)
            else:
                if not args.reverse_complement:
                    full_str = "".join([s.upper() for s in seq_list])
                else:
                    full_str = "".join([RC(s.upper()) for s in seq_list])
                if "N" in full_str:
                    motif_alphabet += "N"
                motif_pwm = np.concatenate((motif_pwm, np.zeros((motif_pwm.shape[0], 1), dtype=np.float64)), axis=1)
                bg_freq = np.array([int(np.char.count(full_str, a)) for a in motif_alphabet], 
                                   dtype=np.float64)
                bg_freq /= np.sum(bg_freq)
            
            total_counts = motif_dataset.get_motif_num_source_sites(motif) 
            total_count_matrix = motif_pwm * total_counts
            motif_pwm = (total_count_matrix + 1) / (total_counts + motif_pwm.shape[1])

            for seq in seq_list:
                strand = "both" if args.reverse_complement else "+"
                motif_score_track = MemeMotif.search_one_motif(seq, 
                                                               motif_alphabet, 
                                                               motif_pwm, 
                                                               bg_freq=bg_freq,
                                                               strand=strand,
                                                               )
                output_track_list.append(motif_score_track)

            input_es.load_region_track_from_list(motif, output_track_list)

            input_es.save_anno_npy(motif, 
                                   args.output_header + "." + motif + ".npy", 
                                   )
