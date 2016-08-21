#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Test H2OFrame.merge() method."""
import h2o
from h2o.exceptions import H2OTypeError, H2OValueError
from tests import pyunit_utils



def test_merge():
    """Test H2OFrame.merge() method."""
    fr = h2o.H2OFrame({"rank": [0, 0, 1, 1, 2, 3, 0] * 10000})
    mapping = h2o.H2OFrame({"rank": [0, 1, 2, 3], "outcome": [6, 7, 8, 9]})

    merged = fr.merge(mapping, all_x=True, all_y=False)
    rows, cols = merged.dim
    assert rows == 70000 and cols == 2, \
        "Expected 70000 rows and 2 cols, but got {0} rows and {1} cols".format(rows, cols)

    threes = merged[merged["rank"] == 3].nrow
    assert threes == 10000, "Expected 10000 3's, but got {0}".format(threes)

    merged2 = fr.merge(mapping, by_x="rank", by_y="outcome")
    assert merged2.nrow == 0, "Expected an empty merge, got %d rows" % merged2.nrow

    merged3 = fr.merge(h2o.H2OFrame({"rank": [0, 10, 100]}), all_x=False, all_y=False)
    assert merged3.nrow == 30000, "Expected 30000 rows, got %d rows" % merged3.nrow

    def merge_error(fr1, fr2, **kwargs):
        try:
            fr1.merge(fr2, **kwargs)
            assert False, "The merge should not have succeeded!"
        except (H2OTypeError, H2OValueError):
            pass

    merge_error(fr, mapping, by_x="rank")
    merge_error(fr, mapping, by_x="r", by_y=0)
    merge_error(fr, mapping, by_x="rank", by_y=3)
    merge_error(fr, mapping, by_x="rank", by_y=["rank", "outcome"])
    merge_error(fr, mapping, by_x=["rank", 0], by_y=["rank", "outcome"])
    merge_error(fr, mapping, by_x=["rank", -1], by_y=["rank", "outcome"])


if __name__ == "__main__":
    pyunit_utils.standalone_test(test_merge)
else:
    test_merge()
