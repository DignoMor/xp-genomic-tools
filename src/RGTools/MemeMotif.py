
import re 
import numpy as np

from .utils import reverse_complement as RC

class MemeMotif:
    def __init__(self, file_path: str = None):
        '''
        Initializes the MemeMotif object with the given file path.

        Keyword arguments:
        - file_path: Path to the MEME motif file. If None, 
        the object will be initialized as an empty MemeMotif object.
        '''
        self.file_path = file_path
        self.version = None
        self.alphabet = None
        self.strands = None
        self.bg_freq = None
        self.motifs = []
        self.motif_info_dict = {}

        if file_path:
            self._parse_meme_file(file_path)
        
    def _parse_meme_file(self, file_path: str):
        '''
        Parses the MEME motif file and extracts relevant information.
        Part of the initialization process.
        
        Keyword arguments:
        - file_path: Path to the MEME motif file.
        '''

        with open(file_path, "r") as f:
            while line := f.readline():
                line = line.strip()
                if line.startswith("MEME version"):
                    self.version = line.split()[-1].strip()
                    break
            
            while line := f.readline():
                line = line.strip()
                if line.startswith("ALPHABET"):
                    self.alphabet = line.split("=")[1].strip()
                    break

            while line := f.readline():
                line = line.strip()
                if line.startswith("strands"):
                    self.strands = line.split()[1:]
                    break

            while line := f.readline():
                line = line.strip()
                if line.startswith("Background letter frequencies"):
                    line = f.readline().strip()
                    line_info = line.split()

                    freq_dict = {}

                    for i in range(len(self.alphabet)):
                        char = line_info[2*i]
                        freq = line_info[2*i + 1]

                        freq_dict[char] = freq
                    
                    self.bg_freq = [float(freq_dict[c]) for c in self.alphabet]
                    break

            while line := f.readline():
                line = line.strip()
                if line.startswith("MOTIF"):
                    motif_name = line.split()[1]
                    motif_info = {}

                    line = f.readline().strip()
                    match = re.match(r"letter-probability matrix: alength=(.*)w=(.*)nsites=(.*)E=(.*)", line)

                    motif_info["alphabet_length"] = int(match.group(1))
                    motif_info["motif_length"] = int(match.group(2))
                    motif_info["num_source_sites"] = int(match.group(3))
                    motif_info["source_eval"] = float(match.group(4))

                    pwm_arr = np.zeros((motif_info["motif_length"], motif_info["alphabet_length"]))
                    for i in range(motif_info["motif_length"]):
                        line = f.readline().strip()
                        line_info = line.split()
                        pwm_arr[i] = np.array(line_info, dtype=float)
                
                    motif_info["pwm"] = pwm_arr

                    self.motifs.append(motif_name)
                    self.motif_info_dict[motif_name] = motif_info

    def write_meme_file(self, file_path: str):
        '''
        Writes the MEME motif file to the given file path.
        '''
        with open(file_path, "w") as f:
            f.write("MEME version {}\n".format(str(self.get_meme_version())))
            f.write("\n")
            f.write("ALPHABET={}\n".format(str(self.get_alphabet())))
            f.write("\n")
            f.write("strands: {}\n".format(" ".join(self.get_strands())))
            f.write("\n")
            f.write("Background letter frequencies\n")
            for alphabet, freq in zip(self.get_alphabet(), self.get_bg_freq()):
                f.write("{} {} ".format(alphabet, freq))
            
            f.write("\n")
            f.write("\n")

            for motif_name in self.get_motif_list():
                f.write("MOTIF {}\n".format(motif_name))
                f.write("letter-probability matrix: alength={} w={} nsites={} E={}\n".format(
                    self.get_motif_alphabet_length(motif_name),
                    self.get_motif_length(motif_name),
                    self.get_motif_num_source_sites(motif_name),
                    self.get_motif_source_eval(motif_name)
                ))
                
                pwm = self.get_motif_pwm(motif_name)
                for i in range(pwm.shape[0]):
                    f.write("\t".join(map(lambda x: "{:.6f}".format(x), pwm[i])))
                    f.write("\n")

                f.write("\n")

    def clone_empty(self):
        '''
        Clones the current MemeMotif object.

        Returns:
        - A new MemeMotif object with the same metadata as self, but 
        no motifs.
        '''
        clone = MemeMotif()
        clone.version = self.version
        clone.alphabet = self.alphabet
        clone.strands = self.strands
        clone.bg_freq = self.bg_freq
        return clone
            
    def get_meme_version(self):
        '''
        Returns the version of the MEME motif file.

        Return:
        - A string representing the version of the MEME motif file.
        '''
        return self.version
    
    def set_meme_version(self, version: str):
        '''
        Sets the version of the MEME motif file.

        Keyword arguments:
        - version: A string representing the version of the MEME motif file.
        '''
        self.version = version

    def get_alphabet(self):
        '''
        Returns the alphabet used in the MEME motif file.

        Return: 
        - A string representing the alphabet used in the MEME motif file.
        '''
        return self.alphabet
    
    def set_alphabet(self, alphabet: str):
        '''
        Sets the alphabet used in the MEME motif file.

        Keyword arguments:
        - alphabet: A string representing the alphabet used in the MEME motif file.
        '''
        self.alphabet = alphabet
    
    def get_strands(self):
        '''
        Returns the strands used in the MEME motif file.

        Return:
        - A list of strings representing the strands used in the MEME motif file.
        '''
        return self.strands
    
    def set_strands(self, strands: list):
        '''
        Sets the strands used in the MEME motif file.

        Keyword arguments:
        - strands: A list of strings representing the strands used in the MEME motif file.
        '''
        self.strands = strands

    def get_bg_freq(self):
        '''
        Returns the background frequency used in the MEME motif file.

        Return: 
        - A list of floats representing background frequencies (same order as alphabet).
        '''
        return self.bg_freq
    
    def set_bg_freq(self, bg_freq: list):
        '''
        Sets the background frequency used in the MEME motif file.

        Keyword arguments:
        - bg_freq: A list of floats representing background frequencies (same order as alphabet).
        '''
        self.bg_freq = bg_freq

    def get_motif_list(self):
        '''
        Returns the list of motifs in the MEME motif file.

        Return: 
        - A list of strings representing the names of the motifs in the MEME motif file.
        '''
        return self.motifs

    def get_motif_pwm(self, motif_name):
        '''
        Returns the position weight matrix (PWM) of the given motif.

        Keyword arguments:
        - motif_name: Name of the motif.

        Return:
        - A 2D numpy array representing the PWM of the given motif.
          The array should have shape (num_positions, num_alphabet_chars).
        
        Raises:
        - KeyError: If motif_name is not found.
        '''
        if motif_name not in self.motif_info_dict:
            raise KeyError(f"Motif '{motif_name}' not found. Available motifs: {self.motifs}")
        return self.motif_info_dict[motif_name]["pwm"]

    def get_motif_alphabet_length(self, motif_name):
        '''
        Returns the length of the alphabet used in the given motif.

        Keyword arguments:
        - motif_name: Name of the motif.

        Return: 
        - An integer representing the length of the alphabet used in the given motif.
        
        Raises:
        - KeyError: If motif_name is not found.
        '''
        if motif_name not in self.motif_info_dict:
            raise KeyError(f"Motif '{motif_name}' not found. Available motifs: {self.motifs}")
        return self.motif_info_dict[motif_name]["alphabet_length"]

    def get_motif_length(self, motif_name):
        '''
        Returns the length of the given motif.

        Keyword arguments:
        - motif_name: Name of the motif.

        Return:
        - An integer representing the length of the given motif.
        
        Raises:
        - KeyError: If motif_name is not found.
        '''
        if motif_name not in self.motif_info_dict:
            raise KeyError(f"Motif '{motif_name}' not found. Available motifs: {self.motifs}")
        return self.motif_info_dict[motif_name]["motif_length"]

    def get_motif_num_source_sites(self, motif_name):
        '''
        Returns the number of source sites for the given motif.

        Keyword arguments:
        - motif_name: Name of the motif.

        Return: 
        - An integer representing the number of source sites for the given motif.
        
        Raises:
        - KeyError: If motif_name is not found.
        '''
        if motif_name not in self.motif_info_dict:
            raise KeyError(f"Motif '{motif_name}' not found. Available motifs: {self.motifs}")
        return self.motif_info_dict[motif_name]["num_source_sites"] 

    def get_motif_source_eval(self, motif_name):
        '''
        Returns the source evaluation for the given motif.

        Keyword arguments:
        - motif_name: Name of the motif.

        Return: 
        - A float representing the source E-value for the given motif.
        
        Raises:
        - KeyError: If motif_name is not found.
        '''
        if motif_name not in self.motif_info_dict:
            raise KeyError(f"Motif '{motif_name}' not found. Available motifs: {self.motifs}")
        return self.motif_info_dict[motif_name]["source_eval"]
    
    def add_motif(self, motif_name: str, motif_info: dict):
        '''
        Adds a motif to the MEME motif file.

        Keyword arguments:
        - motif_name: Name of the motif.
        - motif_info: A dictionary containing the motif information. 
            The dictionary should have the following keys:
            * alphabet_length: An integer representing the length of the alphabet used in the motif.
            * motif_length: An integer representing the length of the motif.
            * num_source_sites: An integer representing the number of source sites for the motif.
            * source_eval: A float representing the source E-value for the motif.
            * pwm: A 2D numpy array representing the PWM of the motif. The 
            array should have shape (motif_length, alphabet_length).
            Rows must be normalized (sum to 1.0).
        '''
        self.motifs.append(motif_name)

        for info_field in ["alphabet_length", "motif_length", "num_source_sites", "source_eval"]:
            if info_field not in motif_info:
                raise ValueError(f"The motif info dictionary must contain the key {info_field}.")

        # Validate PWM exists and has correct shape
        if "pwm" not in motif_info:
            raise ValueError("The motif info dictionary must contain the key 'pwm'.")
        
        pwm = motif_info["pwm"]
        motif_length = motif_info["motif_length"]
        alphabet_length = motif_info["alphabet_length"]
        
        if pwm.shape != (motif_length, alphabet_length):
            raise ValueError(f"PWM shape {pwm.shape} does not match declared dimensions "
                           f"(motif_length={motif_length}, alphabet_length={alphabet_length})")
        
        # Validate PWM normalization (rows should sum to ~1.0)
        row_sums = pwm.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(f"PWM rows must sum to 1.0. Found row sums: {row_sums}")

        self.motif_info_dict[motif_name] = motif_info

    @staticmethod
    def calculate_pwm_score(seq, pwm, alphabet="ACGT", bg_freq=None, reverse_complement=False):
        '''
        Calculate the score of a sequence based on a given position weight matrix (PWM).
        
        Keyword arguments:
        - seq: sequence to score, must be the same length as the pwm.
        - pwm: position weight matrix, a 2D numpy array where each row 
              corresponds to a position in the sequence and each column 
              corresponds to a character in the alphabet. 
              shape = (num_positions, num_alphabet_chars)
        - bg_freq: background frequency of the alphabet, a 1D numpy array
        - reverse_complement: whether to reverse complement the sequence while matching for motifs.

        Returns:
        - score: the score of the sequence based on the PWM.
        '''
        if not len(seq) == pwm.shape[0]:
            raise ValueError("Length of sequence must be the same as the length of the PWM.")
        
        if bg_freq is None: 
            bg_freq = np.ones(len(alphabet)) / len(alphabet)

        if reverse_complement:
            seq = RC(seq)

        alphabet2idx = {char: idx for idx, char in enumerate(alphabet)}
        score = 0

        for i, char in enumerate(seq):
            score += np.log10(pwm[i, alphabet2idx[char]] + 1e-10) - \
                np.log10(bg_freq[alphabet2idx[char]] + 1e-10)

        return score

    @staticmethod
    def search_one_motif(seq, motif_alphabet, motif_pwm, bg_freq=None, strand="+"):
        '''
        Search for a single motif in a sequence.
        Returns array of weighted score based on pwm.

        Keyword arguments:
        - seq: sequence to search
        - motif_alphabet: Alphabet string for the motif
        - motif_pwm: PWM of motif to search for
        - bg_freq: background frequencies (default: uniform)
        - strand: Strand to search. One of "+", "-", or "both".
                  When "both" is given, returned values are the higher score 
                  between forward and reverse complement strand.

        Returns:
        - output_arr: array of scores for each position in the sequence.
                      Array length equals sequence length.
                      output_arr[i] represents matching score of seq[i:i+motif_len].
                      Positions where motif doesn't fit (at the end) are set to minimum score.
        '''
        if strand not in ["+", "-", "both"]:
            raise ValueError(f"strand must be one of '+', '-', or 'both'. Got: {strand}")
        
        seq = seq.upper()

        output_arr = np.zeros(len(seq), 
                              dtype=np.float64, 
                              )

        motif_len = motif_pwm.shape[0]
        seq_len = len(seq)

        for i in range(seq_len - motif_len + 1):
            target_seq = seq[i:i + motif_len]
            
            if strand == "+":
                score = MemeMotif.calculate_pwm_score(target_seq, motif_pwm, 
                                                      motif_alphabet, bg_freq, 
                                                      reverse_complement=False)
                output_arr[i] = score
            elif strand == "-":
                score = MemeMotif.calculate_pwm_score(target_seq, motif_pwm, 
                                                      motif_alphabet, bg_freq, 
                                                      reverse_complement=True)
                output_arr[i] = score
            elif strand == "both":
                score_fwd = MemeMotif.calculate_pwm_score(target_seq, motif_pwm, 
                                                         motif_alphabet, bg_freq, 
                                                         reverse_complement=False)
                score_rev = MemeMotif.calculate_pwm_score(target_seq, motif_pwm, 
                                                         motif_alphabet, bg_freq, 
                                                         reverse_complement=True)
                output_arr[i] = max(score_fwd, score_rev)
        
        # Pad positions where motif doesn't fit with minimum score
        output_arr[-motif_len+1:] = output_arr.min()

        return output_arr
