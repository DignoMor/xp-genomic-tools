
import os
import re
from typing import TextIO, Union

import numpy as np

from .utils import reverse_complement as RC

_PWM_ROW_ATOL = 1e-6
# MEME backgrounds are often rounded to three decimals (e.g. 1.001).
_BACKGROUND_SUM_ATOL = 1e-3
_MATRIX_HEADER_RE = re.compile(
    r"^letter-probability\s+matrix:\s*"
    r"alength\s*=\s*(\S+)\s+"
    r"w\s*=\s*(\S+)\s+"
    r"nsites\s*=\s*(\S+)\s+"
    r"E\s*=\s*(\S+)\s*$",
    re.IGNORECASE,
)


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
            lines = f.readlines()

        index = 0
        n_lines = len(lines)

        def _next_nonempty(start: int):
            i = start
            while i < n_lines:
                raw = lines[i]
                stripped = raw.strip()
                if stripped:
                    return i, stripped
                i += 1
            return n_lines, None

        index, line = _next_nonempty(index)
        if line is None or not line.startswith("MEME version"):
            raise ValueError(
                "MEME file is missing a required 'MEME version' header."
            )
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(
                f"Malformed MEME version header: {line!r}."
            )
        self.version = parts[-1]

        index, line = _next_nonempty(index + 1)
        if line is None or not line.startswith("ALPHABET"):
            raise ValueError(
                "MEME file is missing a required 'ALPHABET=' header."
            )
        if "=" not in line:
            raise ValueError(f"Malformed ALPHABET header: {line!r}.")
        self.alphabet = line.split("=", 1)[1].strip()
        if not self.alphabet:
            raise ValueError("ALPHABET header has an empty alphabet.")

        index, line = _next_nonempty(index + 1)
        if line is None or not line.startswith("strands"):
            raise ValueError(
                "MEME file is missing a required 'strands:' header."
            )
        strand_parts = line.split()
        if len(strand_parts) < 2:
            raise ValueError(f"Malformed strands header: {line!r}.")
        self.strands = strand_parts[1:]

        index, line = _next_nonempty(index + 1)
        if line is None or not line.startswith("Background letter frequencies"):
            raise ValueError(
                "MEME file is missing a required "
                "'Background letter frequencies' header."
            )
        index, line = _next_nonempty(index + 1)
        if line is None:
            raise ValueError(
                "MEME file is missing the background frequency line."
            )
        self.bg_freq = self._parse_background_line(line, self.alphabet)

        index += 1
        while True:
            index, line = _next_nonempty(index)
            if line is None:
                break
            if not line.startswith("MOTIF"):
                raise ValueError(
                    f"Expected a MOTIF header, found: {line!r}."
                )
            motif_parts = line.split()
            if len(motif_parts) < 2:
                raise ValueError(f"Malformed MOTIF header: {line!r}.")
            motif_name = motif_parts[1]
            if motif_name in self.motif_info_dict:
                raise ValueError(
                    f"Duplicate motif name {motif_name!r} in MEME file."
                )

            index, matrix_header = _next_nonempty(index + 1)
            if matrix_header is None:
                raise ValueError(
                    f"Motif {motif_name!r} is missing its "
                    "letter-probability matrix header."
                )
            motif_info = self._parse_matrix_header(matrix_header, motif_name)

            pwm_rows = []
            row_index = index + 1
            for row_i in range(motif_info["motif_length"]):
                row_index, row_line = _next_nonempty(row_index)
                if (
                    row_line is None
                    or row_line.startswith("MOTIF")
                    or row_line.startswith("letter-probability")
                ):
                    raise ValueError(
                        f"Truncated PWM matrix for motif {motif_name!r}: "
                        f"expected {motif_info['motif_length']} rows, "
                        f"found {row_i}."
                    )
                values = row_line.split()
                if len(values) != motif_info["alphabet_length"]:
                    raise ValueError(
                        f"PWM row width mismatch for motif {motif_name!r} "
                        f"at row {row_i}: expected "
                        f"{motif_info['alphabet_length']} columns, "
                        f"found {len(values)}."
                    )
                try:
                    row = np.array(values, dtype=float)
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric PWM values for motif {motif_name!r} "
                        f"at row {row_i}."
                    ) from exc
                pwm_rows.append(row)
                row_index += 1

            pwm_arr = np.vstack(pwm_rows)
            self._validate_pwm(pwm_arr, motif_name)
            motif_info["pwm"] = pwm_arr
            self.motifs.append(motif_name)
            self.motif_info_dict[motif_name] = motif_info
            index = row_index

    @staticmethod
    def _parse_background_line(line: str, alphabet: str):
        tokens = line.split()
        expected = 2 * len(alphabet)
        if len(tokens) != expected:
            raise ValueError(
                "Background frequency line length does not match alphabet: "
                f"expected {expected} tokens for alphabet {alphabet!r}, "
                f"found {len(tokens)}."
            )
        freq_dict = {}
        for i in range(len(alphabet)):
            char = tokens[2 * i]
            freq_token = tokens[2 * i + 1]
            if char != alphabet[i]:
                raise ValueError(
                    "Background frequency letters are misaligned with "
                    f"alphabet {alphabet!r}: expected {alphabet[i]!r}, "
                    f"found {char!r}."
                )
            try:
                freq_dict[char] = float(freq_token)
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric background frequency for letter {char!r}: "
                    f"{freq_token!r}."
                ) from exc

        bg_freq = [freq_dict[c] for c in alphabet]
        arr = np.asarray(bg_freq, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                "Background frequencies must be finite."
            )
        if np.any(arr < 0):
            raise ValueError(
                "Background frequencies must be non-negative."
            )
        if not np.isclose(arr.sum(), 1.0, atol=_BACKGROUND_SUM_ATOL):
            raise ValueError(
                f"Background frequencies must sum to 1.0 within "
                f"atol={_BACKGROUND_SUM_ATOL}; found sum={arr.sum()}."
            )
        return bg_freq

    @staticmethod
    def _parse_matrix_header(line: str, motif_name: str):
        match = _MATRIX_HEADER_RE.match(line.strip())
        if match is None:
            raise ValueError(
                f"Malformed letter-probability matrix header for motif "
                f"{motif_name!r}: {line!r}."
            )
        try:
            alphabet_length = int(match.group(1))
            motif_length = int(match.group(2))
            num_source_sites = int(match.group(3))
            source_eval = float(match.group(4))
        except ValueError as exc:
            raise ValueError(
                f"Invalid numerical metadata in letter-probability matrix "
                f"header for motif {motif_name!r} "
                f"(alength/w/nsites/E): {line!r}."
            ) from exc

        if alphabet_length <= 0 or motif_length <= 0:
            raise ValueError(
                f"Invalid dimensions for motif {motif_name!r}: "
                f"alength={alphabet_length}, w={motif_length}."
            )
        if num_source_sites < 0:
            raise ValueError(
                f"Invalid nsites metadata for motif {motif_name!r}: "
                f"{num_source_sites}."
            )
        if not np.isfinite(source_eval):
            raise ValueError(
                f"Invalid non-finite E-value metadata for motif "
                f"{motif_name!r}."
            )
        return {
            "alphabet_length": alphabet_length,
            "motif_length": motif_length,
            "num_source_sites": num_source_sites,
            "source_eval": source_eval,
        }

    @staticmethod
    def _validate_pwm(pwm, motif_name: str):
        if not np.all(np.isfinite(pwm)):
            raise ValueError(
                f"PWM values for motif {motif_name!r} must be finite."
            )
        if np.any(pwm < 0):
            raise ValueError(
                f"PWM values for motif {motif_name!r} must be non-negative."
            )
        row_sums = pwm.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=_PWM_ROW_ATOL):
            raise ValueError(
                f"PWM rows must sum to 1.0 for motif {motif_name!r} "
                f"(atol={_PWM_ROW_ATOL}). Found row sums: {row_sums}."
            )

    def _format_meme_text(self) -> str:
        chunks = []
        chunks.append("MEME version {}\n".format(str(self.get_meme_version())))
        chunks.append("\n")
        chunks.append("ALPHABET={}\n".format(str(self.get_alphabet())))
        chunks.append("\n")
        chunks.append("strands: {}\n".format(" ".join(self.get_strands())))
        chunks.append("\n")
        chunks.append("Background letter frequencies\n")
        for alphabet, freq in zip(self.get_alphabet(), self.get_bg_freq()):
            chunks.append("{} {} ".format(alphabet, freq))
        chunks.append("\n")
        chunks.append("\n")

        for motif_name in self.get_motif_list():
            chunks.append("MOTIF {}\n".format(motif_name))
            chunks.append(
                "letter-probability matrix: alength={} w={} nsites={} E={}\n".format(
                    self.get_motif_alphabet_length(motif_name),
                    self.get_motif_length(motif_name),
                    self.get_motif_num_source_sites(motif_name),
                    self.get_motif_source_eval(motif_name),
                )
            )
            pwm = self.get_motif_pwm(motif_name)
            for i in range(pwm.shape[0]):
                chunks.append(
                    "\t".join(map(lambda x: "{:.6f}".format(x), pwm[i]))
                )
                chunks.append("\n")
            chunks.append("\n")
        return "".join(chunks)

    def write_meme_file(self, destination: Union[str, os.PathLike, TextIO]):
        '''
        Writes the MEME motif file to a filesystem path or text stream.
        Path and stream destinations emit the same supported-subset text.
        Serialization does not mutate collection metadata or motif arrays.
        '''
        text = self._format_meme_text()
        if hasattr(destination, "write") and not isinstance(
            destination, (str, bytes, os.PathLike)
        ):
            destination.write(text)
            return
        with open(destination, "w") as f:
            f.write(text)

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
        clone.strands = list(self.strands) if self.strands is not None else None
        clone.bg_freq = list(self.bg_freq) if self.bg_freq is not None else None
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
        if motif_name in self.motif_info_dict:
            raise ValueError(f"Duplicate motif name {motif_name!r}.")

        for info_field in ["alphabet_length", "motif_length", "num_source_sites", "source_eval"]:
            if info_field not in motif_info:
                raise ValueError(f"The motif info dictionary must contain the key {info_field}.")

        if "pwm" not in motif_info:
            raise ValueError("The motif info dictionary must contain the key 'pwm'.")

        try:
            alphabet_length = int(motif_info["alphabet_length"])
            motif_length = int(motif_info["motif_length"])
            num_source_sites = int(motif_info["num_source_sites"])
            source_eval = float(motif_info["source_eval"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid numerical metadata for motif {motif_name!r} "
                f"(alphabet_length/motif_length/num_source_sites/source_eval)."
            ) from exc

        if alphabet_length <= 0 or motif_length <= 0:
            raise ValueError(
                f"Invalid dimensions for motif {motif_name!r}: "
                f"alphabet_length={alphabet_length}, motif_length={motif_length}."
            )
        if num_source_sites < 0:
            raise ValueError(
                f"Invalid nsites metadata for motif {motif_name!r}: "
                f"{num_source_sites}."
            )
        if not np.isfinite(source_eval):
            raise ValueError(
                f"Invalid non-finite E-value metadata for motif "
                f"{motif_name!r}."
            )

        pwm = np.asarray(motif_info["pwm"], dtype=float)
        if pwm.shape != (motif_length, alphabet_length):
            raise ValueError(
                f"PWM shape {pwm.shape} does not match declared dimensions "
                f"(motif_length={motif_length}, alphabet_length={alphabet_length})"
            )

        self._validate_pwm(pwm, motif_name)

        stored = {
            "alphabet_length": alphabet_length,
            "motif_length": motif_length,
            "num_source_sites": num_source_sites,
            "source_eval": source_eval,
            "pwm": pwm.copy(),
        }
        self.motifs.append(motif_name)
        self.motif_info_dict[motif_name] = stored

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
