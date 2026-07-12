import json

from scripts.aggregate import aggregate


def _write_rung(base, slug, correct_names, accuracy, readability):
    d = base / slug
    d.mkdir(parents=True)
    (d / "two_hop.json").write_text(json.dumps({
        "baseline_accuracy": accuracy,
        "intermediate_hit_rate_top5": readability,
        "n_items": 90,
        "swap_success_rate_top1": 0.0,
        "swap_success_rate_top5": 0.0,
        "n_swapped": len(correct_names),
        "per_item": [
            {"name": n, "baseline_correct": True,
             "intermediate_band_min_rank": 3, "swap_success_top1": False,
             "swap_success_top5": False}
            for n in correct_names
        ] + [{"name": "miss-x", "baseline_correct": False,
              "intermediate_band_min_rank": 500}],
    }), encoding="utf-8")
    (d / "verbal_report.json").write_text(json.dumps({
        "n_categories": 14, "n_valid_answers": 5,
        "report_hit_rate_top5": 0.2, "report_hit_rate_top5_valid": 0.0,
    }), encoding="utf-8")
    (d / "structure.json").write_text(json.dumps({
        "concentration": {"k": 77, "k_frac": 0.1, "variance_fraction": 0.15},
        "mid_block": [4, 8],
    }), encoding="utf-8")


def test_aggregate_rows_and_matched_subset(tmp_path):
    _write_rung(tmp_path, "gpt2-small", ["a", "b", "c"], 0.08, 0.7)
    _write_rung(tmp_path, "qwen3-0.6b", ["b", "c", "d"], 0.09, 0.9)
    out = aggregate(str(tmp_path))
    assert {r["slug"] for r in out["rungs"]} == {"gpt2-small", "qwen3-0.6b"}
    row = next(r for r in out["rungs"] if r["slug"] == "gpt2-small")
    assert row["params"] == 124_000_000
    assert row["family"] == "gpt2"
    assert abs(row["concentration_ratio"] - 1.5) < 1e-6
    matched = out["capability_matched"]
    assert matched["item_names"] == ["b", "c"]
    assert matched["underpowered"] is True  # < 10 shared items
    assert set(matched["readability_by_slug"]) == {"gpt2-small", "qwen3-0.6b"}
