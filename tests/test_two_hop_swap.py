import json

from jlens import fit
from tests.tiny import TinyDecoder

from jlens_scaling.experiments import two_hop

PROMPTS = ["abcdefghijklmnopqrstuvwxyz" * 4] * 2


def test_two_hop_swap_arm(tmp_path):
    model = TinyDecoder(n_layers=6, d_model=8)
    lens = fit(model, PROMPTS, dim_batch=4)
    data = {
        "items": [
            {
                "name": "toy",
                "category": "geo",
                "prompt": "the capital of the country of the eiffel tower is ",
                "intermediate": "france",
                "swap_to": "japan",
                "answer": "paris",
                "swap_answer": "tokyo",
            }
        ]
    }
    p = tmp_path / "d.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "o.json"
    result = two_hop.run(
        lens, model, str(p), chat=False, out_path=str(out), swap=True
    )
    assert "swap_success_rate_top1" in result
    assert "swap_success_rate_top5" in result
    assert result["swap_scale"] == 1.0
    item = result["per_item"][0]
    if item["baseline_correct"]:
        assert "swap_greedy_token" in item
        assert item["swap_answer_rank"] >= 1
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["experiment"] == "two_hop"


def test_two_hop_without_swap_has_no_swap_keys(tmp_path):
    model = TinyDecoder(n_layers=6, d_model=8)
    lens = fit(model, PROMPTS, dim_batch=4)
    data = {
        "items": [
            {
                "name": "toy",
                "category": "geo",
                "prompt": "the capital of the country of the eiffel tower is ",
                "intermediate": "france",
                "swap_to": "japan",
                "answer": "paris",
                "swap_answer": "tokyo",
            }
        ]
    }
    p = tmp_path / "d.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    result = two_hop.run(
        lens, model, str(p), chat=False, out_path=str(tmp_path / "o.json")
    )
    assert "swap_success_rate_top1" not in result
