"""Tests for pure helper methods in bbpower.power_specter."""
from __future__ import annotations

import pytest

from bbpower.power_specter import BBPowerSpecter


def _make_specter(**attrs: object) -> BBPowerSpecter:
    """Create a BBPowerSpecter without calling __init__."""
    obj = object.__new__(BBPowerSpecter)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


class TestGetMapLabel:
    """Test BBPowerSpecter.get_map_label."""

    def test_basic(self) -> None:
        """Verify 1-indexed band/split naming."""
        ps = _make_specter()
        assert ps.get_map_label(0, 0) == "band1_split1"

    def test_multidigit(self) -> None:
        """Verify correct numbering for higher indices."""
        ps = _make_specter()
        assert ps.get_map_label(2, 3) == "band3_split4"


class TestGetWorkspaceLabel:
    """Test BBPowerSpecter.get_workspace_label."""

    def test_ordered(self) -> None:
        """Canonical ordering puts the smaller index first."""
        ps = _make_specter()
        assert ps.get_workspace_label(0, 1) == "b1_b2"

    def test_reversed(self) -> None:
        """Reversing band order still produces canonical label."""
        ps = _make_specter()
        assert ps.get_workspace_label(2, 0) == "b1_b3"

    def test_same_band(self) -> None:
        """Auto-pair produces repeated index."""
        ps = _make_specter()
        assert ps.get_workspace_label(1, 1) == "b2_b2"


class TestGetFnameWorkspace:
    """Test BBPowerSpecter.get_fname_workspace."""

    def test_contains_prefix(self) -> None:
        """Result is based on prefix_mcm."""
        ps = _make_specter(prefix_mcm="/tmp/mcm_prefix")
        fname = ps.get_fname_workspace(0, 1)
        assert fname.startswith("/tmp/mcm_prefix")
        assert fname.endswith(".fits")


class TestGetCellIterator:
    """Test BBPowerSpecter.get_cell_iterator."""

    def test_count_1band_2splits(self) -> None:
        """1 band, 2 splits -> upper triangle: (0,0), (0,1), (1,1) = 3 pairs."""
        ps = _make_specter(n_bpss=1, nsplits=2)
        items = list(ps.get_cell_iterator())
        assert len(items) == 3

    def test_count_2band_1split(self) -> None:
        """2 bands, 1 split -> band pairs (0,0), (0,1), (1,1) = 3."""
        ps = _make_specter(n_bpss=2, nsplits=1)
        items = list(ps.get_cell_iterator())
        assert len(items) == 3

    def test_tuple_structure(self) -> None:
        """Each yielded item is a 6-tuple of (b1, b2, s1, s2, l1, l2)."""
        ps = _make_specter(n_bpss=1, nsplits=1)
        items = list(ps.get_cell_iterator())
        assert len(items) == 1
        b1, b2, s1, s2, l1, l2 = items[0]
        assert (b1, b2, s1, s2) == (0, 0, 0, 0)
        assert l1 == "band1_split1"
        assert l2 == "band1_split1"
