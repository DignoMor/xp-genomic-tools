
import pandas as pd

from RGTools.ExogeneousSequences import ExogeneousSequences

class ExogeneousSequenceAssemble:
    @staticmethod
    def set_parser(parser):
        """Set up parser for exogeneous sequence assembly operations."""
        subparsers = parser.add_subparsers(dest="operation", required=True)
        
        parser_add_adapter = subparsers.add_parser("add_adapter", 
                                                  help="Add adapter to the exogeneous sequences.")
        ExogeneousSequenceAssemble._set_parser_add_adapter(parser_add_adapter)
        
        parser_concat = subparsers.add_parser("concat", 
                                             help="Concatenate the exogeneous sequences.")
        ExogeneousSequenceAssemble._set_parser_concat(parser_concat)

        parser_barcode = subparsers.add_parser("barcode", 
                                               help="Add barcode to the exogeneous sequences.")
        ExogeneousSequenceAssemble._set_parser_barcode(parser_barcode)

    @staticmethod
    def _set_parser_add_adapter(parser):
        ExogeneousSequences.set_parser_exogeneous_sequences(parser)

        parser.add_argument("--left_adapter_fasta", 
                            help="Path to the 5' adapter fasta file.",
                            required=True,
                            )
        
        parser.add_argument("--right_adapter_fasta", 
                            help="Path to the 3' adapter fasta file.",
                            default=None,
                            )
        
        parser.add_argument("--output_fasta", 
                            help="Path to the output fasta file.",
                            required=True,
                            default=None,
                            )

    @staticmethod
    def _set_parser_concat(parser):
        parser.add_argument("--fasta5", 
                            help="Path to the 5' fasta file.",
                            required=True,
                            )

        parser.add_argument("--fasta3", 
                            help="Path to the 3' fasta file.",
                            required=True,
                            )

        parser.add_argument("--output_fasta", 
                            help="Path to the output fasta file.",
                            required=True,
                            default=None,
                            )
        
        parser.add_argument("--id_method",
                            help="Method to use to generate the id of the output fasta file.",
                            choices=["5", "3", "5_3"], 
                            default="5_3",
                            )

    @staticmethod
    def _set_parser_barcode(parser):
        parser.add_argument("--barcode_fasta", 
                            help="Path to the barcode fasta file.",
                            required=True,
                            )

        parser.add_argument("--input_fasta", 
                            help="Path to the input fasta file. ",
                            required=True,
                            action="append",
                            )
        
        parser.add_argument("--input_class", 
                            help="Class of the input fasta file.",
                            required=True,
                            action="append",
                            )
        
        parser.add_argument("--output_fasta", 
                            help="Path to the output fasta file.",
                            required=True,
                            )
        
        parser.add_argument("--fasta_id_type",
                            help="Type of the fasta id.",
                            choices=["original", "barcode"],
                            default="original",
                            )
                            
        parser.add_argument("--metadata_path",
                            help="Path to the output metadata file.",
                            required=True,
                            )
        
        parser.add_argument("--barcode_method",
                            help="Method to use to add barcode to the exogeneous sequences.",
                            choices=["5", "3", "5_3"],
                            default="5_3",
                            )

    @staticmethod
    def _add_adapter(args):
        if args.left_adapter_fasta:
            left_adapter_seqs = ExogeneousSequences(args.left_adapter_fasta).get_all_region_seqs()
            if len(left_adapter_seqs) != 1:
                raise ValueError("Left adapter fasta file must contain exactly one sequence.")
            left_adapter_seq = left_adapter_seqs[0]
        else:
            left_adapter_seq = ""

        if args.right_adapter_fasta:
            right_adapter_seqs = ExogeneousSequences(args.right_adapter_fasta).get_all_region_seqs()
            if len(right_adapter_seqs) != 1:
                raise ValueError("Right adapter fasta file must contain exactly one sequence.")
            right_adapter_seq = right_adapter_seqs[0]
        else:
            right_adapter_seq = ""

        input_es = ExogeneousSequences(args.fasta)

        output_seq_ids = input_es.get_region_bed_table().get_chrom_names()
        output_seqs = []
        for region_seq in input_es.get_all_region_seqs():
            output_seqs.append(left_adapter_seq + region_seq + right_adapter_seq)

        ExogeneousSequences.write_sequences_to_fasta(output_seq_ids, output_seqs, args.output_fasta)

    @staticmethod
    def _concat(args):
        fasta5_es = ExogeneousSequences(args.fasta5)
        fasta3_es = ExogeneousSequences(args.fasta3)

        output_seqs = []
        output_seq_ids = []
        chrom_names5 = fasta5_es.get_region_bed_table().get_chrom_names()
        chrom_names3 = fasta3_es.get_region_bed_table().get_chrom_names()
        seqs5 = fasta5_es.get_all_region_seqs()
        seqs3 = fasta3_es.get_all_region_seqs()
        for chrom5, chrom3, seq5, seq3 in zip(chrom_names5, chrom_names3, seqs5, seqs3):
            oseq = seq5 + seq3
            if args.id_method == "5":
                oid = chrom5
            elif args.id_method == "3":
                oid = chrom3
            elif args.id_method == "5_3":
                oid = chrom5 + "_" + chrom3
            else:
                raise ValueError(f"Invalid id method: {args.id_method}")
            
            output_seqs.append(oseq)
            output_seq_ids.append(oid)

        ExogeneousSequences.write_sequences_to_fasta(output_seq_ids, output_seqs, args.output_fasta)

    @staticmethod
    def _barcode(args):
        barcode_es = ExogeneousSequences(args.barcode_fasta)

        input_es_list = [ExogeneousSequences(f) for f in args.input_fasta]
        input_elem_nums = [es.get_num_regions() for es in input_es_list]
        barcode_seqs = barcode_es.get_all_region_seqs()

        total_elem_num = sum(input_elem_nums)
        if total_elem_num > barcode_es.get_num_regions():
            raise ValueError("Total number of elements in the input fasta files is greater than the number of regions in the barcode fasta file.")
        
        metadata_df = pd.DataFrame(columns=["barcode", "class", "elem_id", "elem_seq"])
        output_seqs = []
        output_seq_ids = []
        
        for elem_class, input_es in zip(args.input_class, input_es_list):
            used_barcode_list = []
            for elem_seq, elem_id in zip(input_es.get_all_region_seqs(), input_es.get_region_bed_table().get_chrom_names()):
                barcode_seq = barcode_seqs[len(output_seq_ids)]
                if args.barcode_method == "5":
                    oseq = barcode_seq + elem_seq
                elif args.barcode_method == "3":
                    oseq = elem_seq + barcode_seq
                elif args.barcode_method == "5_3":
                    oseq = barcode_seq + elem_seq + barcode_seq
                else:
                    raise ValueError(f"Invalid barcode method: {args.barcode_method}")
                
                output_seqs.append(oseq)
                output_seq_ids.append(elem_id)
                used_barcode_list.append(barcode_seq)

            
            new_metadata = pd.DataFrame({"barcode": used_barcode_list, 
                                         "class": [elem_class] * len(used_barcode_list), 
                                         "elem_id": output_seq_ids[-len(used_barcode_list):], 
                                         "elem_seq": output_seqs[-len(used_barcode_list):], 
                                         })
            metadata_df = pd.concat([metadata_df, new_metadata])

        if args.fasta_id_type == "original":
            fasta_ids = output_seq_ids
        elif args.fasta_id_type == "barcode":
            fasta_ids = [barcode_seq for barcode_seq in barcode_seqs]
        else:
            raise ValueError(f"Invalid fasta id type: {args.fasta_id_type}")
        
        ExogeneousSequences.write_sequences_to_fasta(fasta_ids, 
                                                     output_seqs, 
                                                     args.output_fasta, 
                                                     )
        metadata_df.to_csv(args.metadata_path, 
                           index=False,
                           )

    @staticmethod
    def main(args):
        if args.operation == "add_adapter":
            ExogeneousSequenceAssemble._add_adapter(args)
        elif args.operation == "concat":
            ExogeneousSequenceAssemble._concat(args)
        elif args.operation == "barcode":
            ExogeneousSequenceAssemble._barcode(args)
        else:
            raise ValueError(f"Unknown operation: {args.operation}")
