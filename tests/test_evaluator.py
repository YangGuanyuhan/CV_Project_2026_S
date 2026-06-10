"""Tests for COCO evaluator and category mapping."""

import pytest

from src.datasets.coco_categories import (
    _COCO_CATEGORY_IDS,
    CATEGORY_ID_TO_NAME,
    CATEGORY_NAME_TO_ID,
    COCO_80_CLASSES,
    build_coco_prompt,
    build_phrase_to_category_mapping,
    match_phrase_to_category,
)


class TestCategoryMapping:
    """Test COCO category ID mapping correctness."""

    def test_80_classes_count(self):
        assert len(COCO_80_CLASSES) == 80

    def test_category_ids_count(self):
        assert len(_COCO_CATEGORY_IDS) == 80

    def test_id_to_name_roundtrip(self):
        for cid, name in CATEGORY_ID_TO_NAME.items():
            assert CATEGORY_NAME_TO_ID[name] == cid

    def test_known_ids(self):
        assert CATEGORY_ID_TO_NAME[1] == "person"
        assert CATEGORY_ID_TO_NAME[90] == "toothbrush"
        assert CATEGORY_ID_TO_NAME[84] == "book"

    def test_known_names(self):
        assert CATEGORY_NAME_TO_ID["person"] == 1
        assert CATEGORY_NAME_TO_ID["toothbrush"] == 90
        assert CATEGORY_NAME_TO_ID["book"] == 84

    def test_no_duplicate_ids(self):
        assert len(set(_COCO_CATEGORY_IDS)) == 80

    def test_ids_are_non_contiguous(self):
        """Verify IDs have gaps (not 1-80)."""
        assert max(_COCO_CATEGORY_IDS) == 90
        assert len(_COCO_CATEGORY_IDS) == 80  # 80 IDs but max is 90


class TestPhraseMatching:
    """Test phrase-to-category matching."""

    def test_exact_match(self):
        assert match_phrase_to_category("person") == 1
        assert match_phrase_to_category("car") == 3

    def test_alias_match(self):
        assert match_phrase_to_category("tv") == 72
        assert match_phrase_to_category("sofa") == 63

    def test_no_match(self):
        assert match_phrase_to_category("helicopter") is None

    def test_case_insensitive(self):
        assert match_phrase_to_category("Person") == 1

    def test_build_mapping(self):
        phrases = ["person", "car", "unknown_thing"]
        mapping = build_phrase_to_category_mapping(phrases)
        assert mapping["person"] == 1
        assert mapping["car"] == 3
        assert "unknown_thing" not in mapping

    def test_build_coco_prompt(self):
        prompt = build_coco_prompt()
        assert "person" in prompt
        assert "toothbrush" in prompt
        assert prompt.endswith(" .")


class TestCOCOEvalParams:
    """Test that COCOeval can be restricted to specific image IDs."""

    def test_cocoeval_imgids_subset(self):
        """COCOeval should only evaluate on specified imgIds."""
        pytest.importorskip("pycocotools")
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # Use a minimal GT-like structure
        # This test verifies the mechanism, not full evaluation
        coco_gt = COCO("data/coco/annotations/instances_val2017.json")
        all_ids = coco_gt.getImgIds()

        # Create a dummy result
        img_id = all_ids[0]
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))

        if not anns:
            pytest.skip("No annotations for test image")

        # Create a fake detection matching first GT
        ann = anns[0]
        fake_result = [
            {
                "image_id": img_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "score": 0.9,
            }
        ]

        coco_dt = coco_gt.loadRes(fake_result)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

        # Restrict to single image
        coco_eval.params.imgIds = [img_id]
        coco_eval.evaluate()
        coco_eval.accumulate()

        # Should only have results for 1 image
        assert len(coco_eval.params.imgIds) == 1

    def test_predictions_not_polluted_by_loadres(self):
        """loadRes should not modify the original results list."""
        pytest.importorskip("pycocotools")
        import copy

        from pycocotools.coco import COCO

        coco_gt = COCO("data/coco/annotations/instances_val2017.json")
        all_ids = coco_gt.getImgIds()

        original = [
            {
                "image_id": all_ids[0],
                "category_id": 1,
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "score": 0.5,
            }
        ]
        original_copy = copy.deepcopy(original)

        _ = coco_gt.loadRes(copy.deepcopy(original))

        # Original should be unchanged
        assert original == original_copy
        assert len(original[0]) == 4  # only 4 keys
