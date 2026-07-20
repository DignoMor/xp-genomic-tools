"""Spec-derived tests for RGTools Ensembl SNP client (SPEC002, SPEC009)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from RGTools.SNP_utils import EnsemblRestSearch

REST_GRCH38 = "https://rest.ensembl.org"
REST_GRCH37 = "https://grch37.rest.ensembl.org"


def _json_response(payload):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload
    return resp


def _chromosome_mapping(seq_region_name, start, end, allele_string):
    return {
        "seq_region_name": seq_region_name,
        "start": start,
        "end": end,
        "coord_system": "chromosome",
        "allele_string": allele_string,
    }


def _variation_payload(rsid, start, end, allele_string, chrom="1"):
    return {
        "name": rsid,
        "mappings": [_chromosome_mapping(chrom, start, end, allele_string)],
    }


# ---------------------------------------------------------------------------
# genome_version → REST base URL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "genome_version, expected_url",
    [
        ("hg38", REST_GRCH38),
        ("GRCh38", REST_GRCH38),
        ("hg19", REST_GRCH37),
        ("GRCh37", REST_GRCH37),
    ],
)
def test_supported_genome_version_url_mapping(genome_version, expected_url):
    client = EnsemblRestSearch(genome_version=genome_version)
    assert client.server_url == expected_url
    assert client.genome_version == genome_version


def test_unsupported_genome_version_errors():
    with pytest.raises(Exception, match="not supported"):
        EnsemblRestSearch(genome_version="mm10")


# ---------------------------------------------------------------------------
# get_rsid_from_location: 0-based pos → +1 for Ensembl wire query
# ---------------------------------------------------------------------------


@patch("RGTools.SNP_utils.requests.get")
def test_get_rsid_from_location_converts_0based_pos_and_strips_chr(mock_get):
    mock_get.return_value = _json_response(
        [
            {"id": "rsExact", "start": 101, "end": 101},
            {"id": "rsSpan", "start": 100, "end": 101},
        ]
    )
    client = EnsemblRestSearch(genome_version="hg38")
    rsids = client.get_rsid_from_location("chr1", 100)

    assert rsids == ["rsExact"]
    mock_get.assert_called_once()
    url = mock_get.call_args.args[0]
    assert url.startswith(REST_GRCH38)
    # 0-based pos 100 → Ensembl 1-based closed interval 101-101; chr prefix stripped.
    assert "/overlap/region/human/1:101-101?feature=variation" in url
    assert mock_get.call_args.kwargs["headers"]["Content-Type"] == "application/json"


@patch("RGTools.SNP_utils.requests.get")
def test_get_rsid_from_location_uses_grch37_server(mock_get):
    mock_get.return_value = _json_response([])
    client = EnsemblRestSearch(genome_version="hg19")
    client.get_rsid_from_location("2", 50)

    url = mock_get.call_args.args[0]
    assert url.startswith(REST_GRCH37)
    assert "/overlap/region/human/2:51-51?feature=variation" in url


# ---------------------------------------------------------------------------
# get_rsid_snp_simple_info: BED half-open + UCSC chr prefix
# ---------------------------------------------------------------------------


@patch("RGTools.SNP_utils.requests.get")
def test_get_rsid_snp_simple_info_bed_half_open_and_chr_prefix(mock_get):
    # Ensembl wire: 1-based closed start=end=101, allele A/G
    # SPEC002 simple-info: BED 0-based half-open → start=100, end=101, chrom=chr1
    mock_get.return_value = _json_response(
        _variation_payload("rs123", start=101, end=101, allele_string="A/G")
    )
    client = EnsemblRestSearch(genome_version="hg38")
    info = client.get_rsid_snp_simple_info("rs123")

    assert info == {
        "chrom": "chr1",
        "start": 100,
        "end": 101,
        "bases": "A/G",
    }
    url = mock_get.call_args.args[0]
    assert url.startswith(REST_GRCH38)
    assert "/variation/human/rs123" in url


@patch("RGTools.SNP_utils.requests.get")
def test_get_rsid_snp_simple_info_requires_chromosome_mapping(mock_get):
    mock_get.return_value = _json_response(
        {
            "name": "rsNoChrom",
            "mappings": [
                {
                    "seq_region_name": "GL000",
                    "start": 10,
                    "end": 10,
                    "coord_system": "scaffold",
                    "allele_string": "A/G",
                }
            ],
        }
    )
    client = EnsemblRestSearch(genome_version="hg38")
    with pytest.raises(Exception, match="chromosome"):
        client.get_rsid_snp_simple_info("rsNoChrom")


# ---------------------------------------------------------------------------
# prioritize_rsids: SNP-like selection (allele count among length-1 SNPs)
# ---------------------------------------------------------------------------


@patch("RGTools.SNP_utils.requests.get")
def test_prioritize_rsids_selects_most_allelic_snp(mock_get):
    payloads = {
        "rsIndel": _variation_payload("rsIndel", 101, 105, "A/ATGC"),
        "rsBi": _variation_payload("rsBi", 101, 101, "A/G"),
        "rsMulti": _variation_payload("rsMulti", 101, 101, "A/G/T"),
    }

    def side_effect(url, headers=None):
        for rsid, payload in payloads.items():
            if rsid in url:
                return _json_response(payload)
        return _json_response({})

    mock_get.side_effect = side_effect
    client = EnsemblRestSearch(genome_version="hg38")
    chosen, info = client.prioritize_rsids(["rsIndel", "rsBi", "rsMulti"])

    assert chosen == "rsMulti"
    assert info["bases"] == "A/G/T"
    assert info["chrom"] == "chr1"
    assert info["end"] - info["start"] == 1


@patch("RGTools.SNP_utils.requests.get")
def test_prioritize_rsids_returns_none_when_no_snp(mock_get):
    mock_get.return_value = _json_response(
        _variation_payload("rsIndel", 101, 105, "A/ATGC")
    )
    client = EnsemblRestSearch(genome_version="hg38")
    chosen, info = client.prioritize_rsids(["rsIndel"])
    assert chosen is None
    assert info is None
