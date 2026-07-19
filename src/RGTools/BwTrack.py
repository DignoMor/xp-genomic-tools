import pyBigWig
import sys
import numpy as np
from abc import ABC, abstractmethod

class BaseBwTrack(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def count_single_region(self, chrom, start, end, **kwargs):
        '''
        Count the reads in a single region.
        This is an abstract method that must be implemented by subclasses.
        
        Keyword arguments:
        - chrom: chromosome
        - start: start position
        - end: end position
        '''
        pass

    @staticmethod
    def quantify_signal(signal: np.array, output_type: str):
        '''
        Quantify the signal. For quantification that is 
        not mirror-invariant, the signal should run from 
        the initiation site to the termination site.

        Keyword arguments:
        - signal: signal to quantify
        - output_type: what information is outputted (see get_supported_quantification_type)
        '''
        if output_type == "raw_count":
            return np.sum(signal)
        elif output_type == "RPK":
            return np.sum(signal) / len(signal) * 1e3
        elif output_type == "full_track":
            return signal
        else:
            raise Exception("Unsupported output type ({}).".format(output_type))
    
    @staticmethod
    def get_supported_quantification_type():
        '''
        Return the list of supported quantification types.
        '''
        return ["raw_count", "RPK", "full_track"]

    @staticmethod
    def _process_padding(chrom, start, end, 
                         l_pad, r_pad, min_len_after_padding, 
                         method_resolving_invalid_padding):
        if - l_pad - r_pad + min_len_after_padding > end - start:
            if method_resolving_invalid_padding == "fallback":
                return start, end
            elif method_resolving_invalid_padding == "raise":
                raise Exception("Padding is larger than the region ({}:{:d}-{:d}).".format(chrom, start, end))
            elif method_resolving_invalid_padding == "drop":
                sys.stderr.write("Padding is larger than the region ({}:{:d}-{:d}). Dropping the region.\n".format(chrom, start, end))
                return None, None
            else:
                raise Exception("Unsupported method to resolve invalid padding ({}).".format(method_resolving_invalid_padding))
        return start - l_pad, end + r_pad

class SingleBwTrack(BaseBwTrack):
    def __init__(self, bw_path):
        '''
        Initialize the SingleBwTrack object.
        
        Keyword arguments:
        - bw_path: path to the bigwig file
        '''
        self.bw = pyBigWig.open(bw_path)
    
    def __del__(self):
        '''
        Close the bigwig file.
        '''
        self.bw.close()

    def count_single_region(self, chrom, start, end, 
                            output_type="raw_count", 
                            l_pad=0, r_pad=0, 
                            min_len_after_padding=50,
                            method_resolving_invalid_padding="raise", 
                            **kwargs):
        '''
        Count the reads in a single region.
        Return the counts.

        Keyword arguments:
        - chrom: chromosome
        - start: start position
        - end: end position
        - output_type: what information is outputted (see get_supported_quantification_type)
        - l_pad: left padding
        - r_pad: right padding
        - min_len_after_padding: minimum length of the region after padding
        - method_resolving_invalid_padding: method to resolve invalid padding
        '''
        start, end = self._process_padding(chrom, start, end, l_pad, r_pad, 
                                         min_len_after_padding, method_resolving_invalid_padding)
        if start is None:  # Padding was invalid and method was "drop"
            return np.nan

        if chrom in self.bw.chroms().keys():
            signal = np.nan_to_num(self.bw.values(chrom, start, end))
        else:
            signal = np.zeros(end - start)

        return self.quantify_signal(signal, output_type)

class PairedBwTrack(BaseBwTrack):
    def __init__(self, bw_pl_path, bw_mn_path):
        '''
        Initialize the PairedBwTrack object.
        The mn bw file is assumed to have negative values.
        
        Keyword arguments:
        - bw_pl_path: path to the plus strand bigwig file
        - bw_mn_path: path to the minus strand bigwig file
        '''
        super().__init__()
        self.bw_pl = pyBigWig.open(bw_pl_path)
        self.bw_mn = pyBigWig.open(bw_mn_path)
    
    def __del__(self):
        '''
        Close the bigwig files.
        '''
        self.bw_pl.close()
        self.bw_mn.close()

    def count_single_region(self, chrom, start, end, strand, 
                            output_type="raw_count", 
                            l_pad=0, r_pad=0, 
                            min_len_after_padding=50,
                            method_resolving_invalid_padding="raise", 
                            flip_mn=False,
                            negative_mn=True,
                            **kwargs):
        '''
        Count the reads in a single region.
        Return the counts.

        Keyword arguments:
        - chrom: chromosome
        - start: start position
        - end: end position
        - strand: strandness, "+" or "-" or ".". If ".", sum of quantifications of both strands will be outputted.
        - output_type: what information is outputted (see get_supported_quantification_type)
        - l_pad: left padding
        - r_pad: right padding
        - min_len_after_padding: minimum length of the region after padding
        - method_resolving_invalid_padding: method to resolve invalid padding
        - flip_mn: if to flip the minus strand signal
        - negative_mn: whether to output the minus strand signal as negative
        '''
        start, end = self._process_padding(chrom, start, end, l_pad, r_pad, 
                                         min_len_after_padding, method_resolving_invalid_padding)
        if start is None:  # Padding was invalid and method was "drop"
            return np.nan

        if chrom in self.bw_pl.chroms().keys():
            pl_sig = np.nan_to_num(self.bw_pl.values(chrom, start, end))
        else:
            pl_sig = np.zeros(end - start)

        if chrom in self.bw_mn.chroms().keys():
            mn_sig = np.abs(np.nan_to_num(self.bw_mn.values(chrom, start, end)))

            if negative_mn:
                mn_sig = -mn_sig

            if flip_mn:
                mn_sig = np.flip(mn_sig)
        else:
            mn_sig = np.zeros(end - start)

        if strand == "+":
            return self.quantify_signal(pl_sig, output_type)
        elif strand == "-":
            return self.quantify_signal(mn_sig, output_type)
        elif strand == ".":
            return (self.quantify_signal(pl_sig, output_type) + 
                   self.quantify_signal(mn_sig, output_type))
        else:
            raise Exception("Invalid strand type.")

