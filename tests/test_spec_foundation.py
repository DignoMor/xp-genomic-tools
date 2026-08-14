"""Spec-derived tests for RGTools foundation API (SPEC002 ListFile + SPEC003)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from RGTools.exceptions import (
    BedTableException,
    BedTableLoadException,
    GTFHandleFilterException,
    GTFRecordNoFeatureException,
    InvalidBedRegionException,
    InvalidStrandnessException,
    RGToolsInternalException,
)
from RGTools.ListFile import ListFile
from RGTools.logging import Logger
from RGTools.utils import NumpyEncoder, reverse_complement, str2bool, str2none

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spec"


# ---------------------------------------------------------------------------
# Exception hierarchy (SPEC003)
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_root_is_exception(self):
        assert issubclass(RGToolsInternalException, Exception)

    @pytest.mark.parametrize(
        "exc",
        [
            GTFHandleFilterException,
            GTFRecordNoFeatureException,
            BedTableException,
        ],
    )
    def test_direct_children_of_root(self, exc):
        assert issubclass(exc, RGToolsInternalException)

    @pytest.mark.parametrize(
        "exc",
        [
            BedTableLoadException,
            InvalidBedRegionException,
            InvalidStrandnessException,
        ],
    )
    def test_bedtable_subtree(self, exc):
        assert issubclass(exc, BedTableException)
        assert issubclass(exc, RGToolsInternalException)

    def test_construct_with_message(self):
        err = BedTableLoadException("bad load")
        assert "bad load" in str(err)


# ---------------------------------------------------------------------------
# Logger (SPEC003)
# ---------------------------------------------------------------------------


class TestLogger:
    def test_indent_prefixes_message_on_stderr(self):
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            log = Logger(opath="stderr", indent_level=0, indentation="  ")
            log.take_log("root")
            log.indent()
            log.take_log("child")
            log.unindent()
            log.take_log("root-again")
        finally:
            sys.stderr = old

        assert buf.getvalue() == "root\n  child\nroot-again\n"

    def test_unindent_below_zero_raises_value_error(self):
        log = Logger(indent_level=0)
        with pytest.raises(ValueError):
            log.unindent()

    def test_path_mode_appends_per_write(self, tmp_path):
        path = tmp_path / "log.txt"
        log = Logger(opath=str(path), indent_level=1, indentation="\t")
        log.take_log("first")
        log.take_log("second")
        assert path.read_text() == "\tfirst\n\tsecond\n"

    def test_stdout_opath(self):
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            Logger(opath="stdout").take_log("hi")
        finally:
            sys.stdout = old
        assert buf.getvalue() == "hi\n"


# ---------------------------------------------------------------------------
# Utils (SPEC003)
# ---------------------------------------------------------------------------


class TestStr2Bool:
    @pytest.mark.parametrize(
        "token", ["", "FALSE", "false", "False", "NONE", "none", "None"]
    )
    def test_falsy_tokens(self, token):
        assert str2bool(token) is False

    @pytest.mark.parametrize("token", ["TRUE", "yes", "0", "1", "anything"])
    def test_other_strings_are_true(self, token):
        assert str2bool(token) is True

class TestStr2None:
    @pytest.mark.parametrize("token", ["NONE", "none", "None"])
    def test_none_token(self, token):
        assert str2none(token) is None

    def test_passthrough(self):
        assert str2none("abc") == "abc"
        assert str2none("") == ""


class TestReverseComplement:
    def test_default_mapping_upper_and_lower(self):
        assert reverse_complement("ATGCN") == "NGCAT"
        assert reverse_complement("atgcn") == "ngcat"

    def test_unmapped_base_raises_key_error(self):
        with pytest.raises(KeyError):
            reverse_complement("ATGX")

    def test_custom_mapping(self):
        assert reverse_complement("AB", mapping={"A": "X", "B": "Y"}) == "YX"


class TestNumpyEncoder:
    def test_ndarray_and_scalar(self):
        payload = {"arr": np.array([1, 2, 3]), "x": np.float64(1.5)}
        assert json.loads(json.dumps(payload, cls=NumpyEncoder)) == {
            "arr": [1, 2, 3],
            "x": 1.5,
        }


# ---------------------------------------------------------------------------
# ListFile (SPEC002 + SPEC003)
# ---------------------------------------------------------------------------


class TestListFile:
    def test_read_filters_empty_and_whitespace_by_default(self):
        lf = ListFile(filter_empty_lines=True)
        lf.read_file(str(FIXTURES / "sample.list"))
        assert list(lf.get_contents()) == ["alpha", "beta", "gamma"]
        assert lf.get_num_lines() == 3

    def test_read_keeps_blank_lines_when_filter_disabled(self):
        lf = ListFile(filter_empty_lines=False)
        lf.read_file(str(FIXTURES / "sample.list"))
        contents = list(lf.get_contents())
        assert contents[0] == "alpha"
        assert "" in contents
        assert any(line.strip() == "" and line != "" for line in contents) or "  " in contents
        assert contents[-1] == "gamma"
        assert lf.get_num_lines() == 5

    def test_write_list_to_file_roundtrip(self, tmp_path):
        out = tmp_path / "out.list"
        ListFile.write_list_to_file(["one", "two"], str(out))
        assert out.read_text() == "one\ntwo\n"

        lf = ListFile()
        lf.read_file(str(out))
        assert list(lf.get_contents()) == ["one", "two"]

    def test_get_contents_dtype(self, tmp_path):
        path = tmp_path / "nums.list"
        path.write_text("1\n2\n3\n")
        lf = ListFile()
        lf.read_file(str(path))
        vals = lf.get_contents(dtype="int")
        assert isinstance(vals, np.ndarray)
        assert vals.dtype.kind in ("i", "u")
        assert vals.tolist() == [1, 2, 3]

    def test_empty_until_read(self):
        lf = ListFile()
        assert lf.get_num_lines() == 0

    def test_stdin_aliases(self):
        for alias in ("stdin", "-"):
            buf = io.StringIO("a\n\nb\n")
            old = sys.stdin
            sys.stdin = buf
            try:
                lf = ListFile(filter_empty_lines=True)
                lf.read_file(alias)
                assert list(lf.get_contents()) == ["a", "b"]
            finally:
                sys.stdin = old

    def test_stdout_aliases(self):
        for alias in ("stdout", "-"):
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                ListFile.write_list_to_file(["x", "y"], alias)
            finally:
                sys.stdout = old
            assert buf.getvalue() == "x\ny\n"
