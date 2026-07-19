
from RGTools.ExogeneousSequences import ExogeneousSequences

class Mutagenesis:
    @staticmethod
    def set_parser_mutagenesis(parser):
        ExogeneousSequences.set_parser_exogeneous_sequences(parser)

        parser.add_argument("--loc_npy",
                            help="Path to the location npy file.",
                            required=True,
                            )
        
        parser.add_argument("--mut_fasta",
                            help="Path to the mutation target fasta file.",
                            required=True,
                            )

        parser.add_argument("--output_fasta",
                            help="Path to the output fasta file.",
                            required=True,
                            )
    
    @staticmethod
    def mutagenesis_main(args):

        input_es = ExogeneousSequences(args.fasta)
        target_es = ExogeneousSequences(args.mut_fasta)

        input_es.load_region_anno_from_npy("loc", args.loc_npy, anno_type="stat")
        mut_locs = input_es.get_stat_arr("loc")

        output_seqs = []
        output_seq_ids = []

        all_elem_seqs = input_es.get_all_region_seqs()
        all_elem_ids = input_es.get_region_bed_table().get_chrom_names()
        all_target_seqs = target_es.get_all_region_seqs()
        all_target_ids = target_es.get_region_bed_table().get_chrom_names()
        if input_es.get_num_regions() == target_es.get_num_regions():
            # elem-wise operation
            for i in range(input_es.get_num_regions()):
                elem_seq = all_elem_seqs[i]
                target_seq = all_target_seqs[i]
                target_len = len(target_seq)

                output_seq = elem_seq[:mut_locs[i, 0]] + target_seq + elem_seq[mut_locs[i, 0] + target_len:]
                output_seqs.append(output_seq)
                output_seq_ids.append(all_elem_ids[i] + "_mut_" + all_target_ids[i])
        else:
            # broadcast operation
            for target_seq_id, target_seq in zip(all_target_ids, all_target_seqs):
                for i in range(input_es.get_num_regions()):
                    elem_seq = all_elem_seqs[i]
                    target_len = len(target_seq)

                    output_seq = elem_seq[:mut_locs[i, 0]] + target_seq + elem_seq[mut_locs[i, 0] + target_len:]
                    output_seqs.append(output_seq)
                    output_seq_ids.append(all_elem_ids[i] + "_mut_" + target_seq_id)
            
        ExogeneousSequences.write_sequences_to_fasta(output_seq_ids, output_seqs, args.output_fasta)
