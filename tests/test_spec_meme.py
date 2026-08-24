"""SPEC002 / SPEC006 black-box tests for MemeMotif (MEME subset)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from RGTools import MemeMotif

FIXTURES_SPEC = Path(__file__).resolve().parent / "fixtures" / "spec"
TINY_MEME = FIXTURES_SPEC / "tiny.meme"

VALID_MEME_HEADER = """MEME version 4

ALPHABET=ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

"""


def _write_meme(tmp_path: Path, body: str, name: str = "case.meme") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _motif_block(
    name: str = "SPEC_TINY",
    rows: list[str] | None = None,
    header: str | None = None,
) -> str:
    if header is None:
        header = "letter-probability matrix: alength= 4 w= 3 nsites= 8 E= 1e-4"
    if rows is None:
        rows = [
            "0.700000\t0.100000\t0.100000\t0.100000",
            "0.100000\t0.700000\t0.100000\t0.100000",
            "0.100000\t0.100000\t0.700000\t0.100000",
        ]
    return f"MOTIF {name}\n{header}\n" + "\n".join(rows) + "\n"

# Expected PWM for SPEC_TINY in the vendored fixture (rows sum to 1).
EXPECTED_PWM = np.array(
    [
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
    ],
    dtype=float,
)


def test_parse_vendored_meme_subset():
    mm = MemeMotif(str(TINY_MEME))
    assert mm.get_meme_version() == "4"
    assert mm.get_alphabet() == "ACGT"
    assert mm.get_strands() == ["+", "-"]
    assert mm.get_bg_freq() == [0.25, 0.25, 0.25, 0.25]
    assert mm.get_motif_list() == ["SPEC_TINY"]
    assert mm.get_motif_length("SPEC_TINY") == 3
    assert mm.get_motif_alphabet_length("SPEC_TINY") == 4
    assert mm.get_motif_num_source_sites("SPEC_TINY") == 8
    np.testing.assert_allclose(mm.get_motif_pwm("SPEC_TINY"), EXPECTED_PWM, atol=1e-6)


def test_write_read_round_trip(tmp_path):
    mm = MemeMotif(str(TINY_MEME))
    out = tmp_path / "roundtrip.meme"
    mm.write_meme_file(str(out))

    reloaded = MemeMotif(str(out))
    assert reloaded.get_meme_version() == mm.get_meme_version()
    assert reloaded.get_alphabet() == mm.get_alphabet()
    assert reloaded.get_strands() == mm.get_strands()
    assert reloaded.get_bg_freq() == mm.get_bg_freq()
    assert reloaded.get_motif_list() == mm.get_motif_list()
    np.testing.assert_allclose(
        reloaded.get_motif_pwm("SPEC_TINY"),
        mm.get_motif_pwm("SPEC_TINY"),
        atol=1e-6,
    )
    assert reloaded.get_motif_num_source_sites("SPEC_TINY") == mm.get_motif_num_source_sites(
        "SPEC_TINY"
    )
    assert reloaded.get_motif_source_eval("SPEC_TINY") == mm.get_motif_source_eval("SPEC_TINY")


def test_add_motif_rejects_unnormalized_pwm_rows():
    mm = MemeMotif()
    mm.set_meme_version("4")
    mm.set_alphabet("ACGT")
    mm.set_strands(["+", "-"])
    mm.set_bg_freq([0.25, 0.25, 0.25, 0.25])

    bad_pwm = EXPECTED_PWM.copy()
    bad_pwm[0, 0] = 0.9  # row sum ≠ 1
    motif_info = {
        "alphabet_length": 4,
        "motif_length": 3,
        "num_source_sites": 4,
        "source_eval": 0.01,
        "pwm": bad_pwm,
    }
    with pytest.raises(ValueError):
        mm.add_motif("BAD_ROW", motif_info)


def test_add_motif_accepts_normalized_pwm():
    mm = MemeMotif()
    mm.set_meme_version("4")
    mm.set_alphabet("ACGT")
    mm.set_strands(["+", "-"])
    mm.set_bg_freq([0.25, 0.25, 0.25, 0.25])

    motif_info = {
        "alphabet_length": 4,
        "motif_length": 3,
        "num_source_sites": 4,
        "source_eval": 0.01,
        "pwm": EXPECTED_PWM.copy(),
    }
    mm.add_motif("OK", motif_info)
    assert mm.get_motif_list() == ["OK"]
    np.testing.assert_allclose(mm.get_motif_pwm("OK"), EXPECTED_PWM, atol=1e-6)


def test_calculate_pwm_score_length_mismatch():
    with pytest.raises(ValueError):
        MemeMotif.calculate_pwm_score("AC", EXPECTED_PWM)


def test_calculate_pwm_score_matching_length():
    score = MemeMotif.calculate_pwm_score("ACG", EXPECTED_PWM)
    assert isinstance(score, (float, np.floating))
    assert np.isfinite(score)


def test_search_one_motif_strand_plus_minus_both():
    seq = "TTACGTTT"
    alphabet = "ACGT"
    scores_plus = MemeMotif.search_one_motif(seq, alphabet, EXPECTED_PWM, strand="+")
    scores_minus = MemeMotif.search_one_motif(seq, alphabet, EXPECTED_PWM, strand="-")
    scores_both = MemeMotif.search_one_motif(seq, alphabet, EXPECTED_PWM, strand="both")

    assert len(scores_plus) == len(seq)
    assert len(scores_minus) == len(seq)
    assert len(scores_both) == len(seq)

    # Forward and reverse strands are not identical for this sequence/PWM.
    assert not np.allclose(scores_plus, scores_minus)

    # Spec: `both` takes the max of forward and RC over scorable windows.
    motif_len = EXPECTED_PWM.shape[0]
    n_windows = len(seq) - motif_len + 1
    np.testing.assert_allclose(
        scores_both[:n_windows],
        np.maximum(scores_plus[:n_windows], scores_minus[:n_windows]),
        atol=1e-9,
    )


def test_search_one_motif_invalid_strand():
    with pytest.raises(ValueError):
        MemeMotif.search_one_motif("ACGTTT", "ACGT", EXPECTED_PWM, strand="x")


# --- SPEC006 supported-subset hardening ---


def test_spec006_write_meme_file_path_and_stream_match(tmp_path):
    mm = MemeMotif(str(TINY_MEME))
    path_out = tmp_path / "path.meme"
    mm.write_meme_file(str(path_out))
    path_text = path_out.read_text(encoding="utf-8")

    stream = io.StringIO()
    mm.write_meme_file(stream)
    stream_text = stream.getvalue()

    assert stream_text == path_text
    assert "0.700000" in path_text
    assert path_text.count(".") >= 6  # six-decimal PWM serialization present


def test_spec006_write_does_not_mutate_pwm_or_metadata(tmp_path):
    mm = MemeMotif(str(TINY_MEME))
    pwm_before = mm.get_motif_pwm("SPEC_TINY").copy()
    motifs_before = list(mm.get_motif_list())
    bg_before = list(mm.get_bg_freq())

    mm.write_meme_file(str(tmp_path / "out.meme"))
    stream = io.StringIO()
    mm.write_meme_file(stream)

    np.testing.assert_array_equal(mm.get_motif_pwm("SPEC_TINY"), pwm_before)
    assert mm.get_motif_list() == motifs_before
    assert mm.get_bg_freq() == bg_before


def test_spec006_parse_rejects_duplicate_motif_names(tmp_path):
    body = VALID_MEME_HEADER + _motif_block("DUP") + _motif_block("DUP")
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match="(?i)duplicate|DUP"):
        MemeMotif(str(path))


def test_spec006_parse_rejects_malformed_matrix_header(tmp_path):
    body = (
        VALID_MEME_HEADER
        + "MOTIF BAD\n"
        + "letter-probability matrix: broken\n"
        + "0.25 0.25 0.25 0.25\n"
    )
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match="(?i)letter-probability|matrix|header"):
        MemeMotif(str(path))


def test_spec006_parse_rejects_missing_meme_version(tmp_path):
    body = """ALPHABET=ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

""" + _motif_block()
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match="(?i)MEME version"):
        MemeMotif(str(path))


def test_spec006_parse_rejects_truncated_matrix(tmp_path):
    body = (
        VALID_MEME_HEADER
        + _motif_block(
            rows=[
                "0.700000\t0.100000\t0.100000\t0.100000",
                "0.100000\t0.700000\t0.100000\t0.100000",
            ]
        )
    )
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match="(?i)truncat|row|matrix|motif"):
        MemeMotif(str(path))


def test_spec006_parse_rejects_wrong_row_width(tmp_path):
    body = (
        VALID_MEME_HEADER
        + _motif_block(
            rows=[
                "0.700000\t0.100000\t0.100000\t0.100000",
                "0.100000\t0.700000\t0.200000",
                "0.100000\t0.100000\t0.700000\t0.100000",
            ]
        )
    )
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match="(?i)width|column|alphabet|row"):
        MemeMotif(str(path))


def test_spec006_parse_rejects_unnormalized_pwm_row(tmp_path):
    body = (
        VALID_MEME_HEADER
        + _motif_block(
            rows=[
                "0.900000\t0.100000\t0.100000\t0.100000",
                "0.100000\t0.700000\t0.100000\t0.100000",
                "0.100000\t0.100000\t0.700000\t0.100000",
            ]
        )
    )
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match="(?i)sum|normal"):
        MemeMotif(str(path))


def test_spec006_parse_rejects_negative_pwm_value(tmp_path):
    body = (
        VALID_MEME_HEADER
        + _motif_block(
            rows=[
                "-0.100000\t0.700000\t0.200000\t0.200000",
                "0.100000\t0.700000\t0.100000\t0.100000",
                "0.100000\t0.100000\t0.700000\t0.100000",
            ]
        )
    )
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match="(?i)negative|non-?negative|PWM"):
        MemeMotif(str(path))


def test_spec006_parse_rejects_nonfinite_pwm_value(tmp_path):
    body = (
        VALID_MEME_HEADER
        + _motif_block(
            rows=[
                "nan\t0.333333\t0.333333\t0.333334",
                "0.100000\t0.700000\t0.100000\t0.100000",
                "0.100000\t0.100000\t0.700000\t0.100000",
            ]
        )
    )
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match="(?i)finite|nan|PWM"):
        MemeMotif(str(path))


def test_spec006_parse_rejects_invalid_background_sum(tmp_path):
    body = """MEME version 4

ALPHABET=ACGT

strands: + -

Background letter frequencies
A 0.5 C 0.5 G 0.5 T 0.5

""" + _motif_block()
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match="(?i)background|sum|normal"):
        MemeMotif(str(path))


def test_spec006_parse_rejects_negative_background(tmp_path):
    body = """MEME version 4

ALPHABET=ACGT

strands: + -

Background letter frequencies
A 0.5 C 0.5 G 0.25 T -0.25

""" + _motif_block()
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match="(?i)background|negative"):
        MemeMotif(str(path))


def test_spec006_parse_rejects_invalid_nsites_metadata(tmp_path):
    body = (
        VALID_MEME_HEADER
        + _motif_block(
            header="letter-probability matrix: alength= 4 w= 3 nsites= notanumber E= 1e-4"
        )
    )
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match="(?i)nsites|metadata"):
        MemeMotif(str(path))


def test_spec006_add_motif_rejects_duplicate_name_and_negative_pwm():
    mm = MemeMotif()
    mm.set_meme_version("4")
    mm.set_alphabet("ACGT")
    mm.set_strands(["+", "-"])
    mm.set_bg_freq([0.25, 0.25, 0.25, 0.25])
    motif_info = {
        "alphabet_length": 4,
        "motif_length": 3,
        "num_source_sites": 4,
        "source_eval": 0.01,
        "pwm": EXPECTED_PWM.copy(),
    }
    mm.add_motif("OK", motif_info)
    with pytest.raises(ValueError, match="(?i)duplicate|OK"):
        mm.add_motif("OK", motif_info)

    bad = EXPECTED_PWM.copy()
    bad[0, 0] = -0.1
    bad[0, 1] = 0.9
    motif_info_neg = {
        "alphabet_length": 4,
        "motif_length": 3,
        "num_source_sites": 4,
        "source_eval": 0.01,
        "pwm": bad,
    }
    with pytest.raises(ValueError, match="(?i)negative|non-?negative|PWM"):
        mm.add_motif("NEG", motif_info_neg)


def test_spec006_add_motif_rejects_invalid_dimensions_and_eval():
    mm = MemeMotif()
    mm.set_meme_version("4")
    mm.set_alphabet("ACGT")
    mm.set_strands(["+", "-"])
    mm.set_bg_freq([0.25, 0.25, 0.25, 0.25])
    with pytest.raises(ValueError, match="(?i)dimension|alphabet_length|motif_length"):
        mm.add_motif(
            "BAD_DIM",
            {
                "alphabet_length": 0,
                "motif_length": 3,
                "num_source_sites": 4,
                "source_eval": 0.01,
                "pwm": EXPECTED_PWM.copy(),
            },
        )
    with pytest.raises(ValueError, match="(?i)E-value|finite|metadata"):
        mm.add_motif(
            "BAD_E",
            {
                "alphabet_length": 4,
                "motif_length": 3,
                "num_source_sites": 4,
                "source_eval": float("nan"),
                "pwm": EXPECTED_PWM.copy(),
            },
        )


def test_spec006_parse_rejects_invalid_alength_dimension(tmp_path):
    body = (
        VALID_MEME_HEADER
        + _motif_block(
            header="letter-probability matrix: alength= 0 w= 3 nsites= 8 E= 1e-4"
        )
    )
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match="(?i)dimension|alength|w"):
        MemeMotif(str(path))


# --- SPEC002 / SPEC006 optional per-motif URL record ---


def test_spec002_006_parse_accepts_trailing_url_record(tmp_path):
    body = (
        VALID_MEME_HEADER
        + _motif_block()
        + "URL http://example.org/MA0139.2\n"
    )
    path = _write_meme(tmp_path, body)
    mm = MemeMotif(str(path))
    assert mm.get_motif_list() == ["SPEC_TINY"]
    np.testing.assert_allclose(mm.get_motif_pwm("SPEC_TINY"), EXPECTED_PWM, atol=1e-6)


def test_spec002_006_parse_accepts_url_between_motifs(tmp_path):
    body = (
        VALID_MEME_HEADER
        + _motif_block("FIRST")
        + "URL http://example.org/first\n"
        + _motif_block("SECOND")
    )
    path = _write_meme(tmp_path, body)
    mm = MemeMotif(str(path))
    assert mm.get_motif_list() == ["FIRST", "SECOND"]
    np.testing.assert_allclose(mm.get_motif_pwm("FIRST"), EXPECTED_PWM, atol=1e-6)
    np.testing.assert_allclose(mm.get_motif_pwm("SECOND"), EXPECTED_PWM, atol=1e-6)


def _assert_modeled_collections_equal(left: MemeMotif, right: MemeMotif) -> None:
    assert left.get_meme_version() == right.get_meme_version()
    assert left.get_alphabet() == right.get_alphabet()
    assert left.get_strands() == right.get_strands()
    assert left.get_bg_freq() == right.get_bg_freq()
    assert left.get_motif_list() == right.get_motif_list()
    for name in left.get_motif_list():
        assert left.get_motif_alphabet_length(name) == right.get_motif_alphabet_length(name)
        assert left.get_motif_length(name) == right.get_motif_length(name)
        assert left.get_motif_num_source_sites(name) == right.get_motif_num_source_sites(name)
        assert left.get_motif_source_eval(name) == right.get_motif_source_eval(name)
        np.testing.assert_allclose(
            left.get_motif_pwm(name),
            right.get_motif_pwm(name),
            atol=1e-6,
        )


def test_spec002_006_url_and_non_url_collections_are_modeled_equal(tmp_path):
    without_url = VALID_MEME_HEADER + _motif_block("A") + _motif_block("B")
    with_url = (
        VALID_MEME_HEADER
        + _motif_block("A")
        + "URL http://example.org/A\n"
        + _motif_block("B")
        + "URL http://example.org/B\n"
    )
    left = MemeMotif(str(_write_meme(tmp_path, without_url, "without.meme")))
    right = MemeMotif(str(_write_meme(tmp_path, with_url, "with.meme")))
    _assert_modeled_collections_equal(left, right)


def test_spec002_006_url_and_non_url_serialize_identically_without_url(tmp_path):
    without_url = VALID_MEME_HEADER + _motif_block("A") + _motif_block("B")
    with_url = (
        VALID_MEME_HEADER
        + _motif_block("A")
        + "URL http://example.org/A\n"
        + _motif_block("B")
        + "URL http://example.org/B\n"
    )
    left = MemeMotif(str(_write_meme(tmp_path, without_url, "without.meme")))
    right = MemeMotif(str(_write_meme(tmp_path, with_url, "with.meme")))
    left_out = io.StringIO()
    right_out = io.StringIO()
    left.write_meme_file(left_out)
    right.write_meme_file(right_out)
    assert left_out.getvalue() == right_out.getvalue()
    assert "URL" not in right_out.getvalue()


@pytest.mark.parametrize(
    ("url_line", "match"),
    [
        ("URL\n", "(?i)malformed|URL|SPEC_TINY"),
        ("URL http://example.org/a http://example.org/b\n", "(?i)malformed|URL|SPEC_TINY"),
        ("url http://example.org/a\n", "(?i)MOTIF|url"),
        ("URLfoo http://example.org/a\n", "(?i)MOTIF|URLfoo"),
    ],
)
def test_spec002_006_parse_rejects_invalid_url_grammar(tmp_path, url_line, match):
    body = VALID_MEME_HEADER + _motif_block() + url_line
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match=match):
        MemeMotif(str(path))


def test_spec002_006_parse_rejects_duplicate_url_for_motif(tmp_path):
    body = (
        VALID_MEME_HEADER
        + _motif_block()
        + "URL http://example.org/a\n"
        + "URL http://example.org/b\n"
    )
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match="(?i)duplicate.*URL.*SPEC_TINY|URL.*duplicate.*SPEC_TINY"):
        MemeMotif(str(path))


def test_spec002_006_parse_rejects_url_before_first_motif(tmp_path):
    body = VALID_MEME_HEADER + "URL http://example.org/x\n" + _motif_block()
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match="(?i)URL.*(before any motif|placement|precedes)"):
        MemeMotif(str(path))


def test_spec002_006_url_inside_incomplete_pwm_is_matrix_failure(tmp_path):
    body = (
        VALID_MEME_HEADER
        + "MOTIF SPEC_TINY\n"
        + "letter-probability matrix: alength= 4 w= 3 nsites= 8 E= 1e-4\n"
        + "0.700000\t0.100000\t0.100000\t0.100000\n"
        + "URL http://example.org/mid\n"
        + "0.100000\t0.100000\t0.700000\t0.100000\n"
    )
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match="(?i)truncat|matrix|PWM|row|numeric|SPEC_TINY"):
        MemeMotif(str(path))


def test_spec002_006_parse_rejects_unrelated_post_motif_record(tmp_path):
    body = VALID_MEME_HEADER + _motif_block() + "COMMAND skip this\n"
    path = _write_meme(tmp_path, body)
    with pytest.raises(ValueError, match="(?i)MOTIF|COMMAND"):
        MemeMotif(str(path))
