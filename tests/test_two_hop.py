import json

from jlens import fit
from tests.tiny import TinyDecoder

from jlens_scaling.experiments import two_hop

PROMPTS = ["abcdefghijklmnopqrstuvwxyz" * 4] * 2


def test_two_hop_runs_on_tiny(tmp_path):
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
    data_path = tmp_path / "probe.json"
    data_path.write_text(json.dumps(data), encoding="utf-8")
    out_path = tmp_path / "out.json"
    result = two_hop.run(
        lens, model, str(data_path), chat=False, out_path=str(out_path)
    )
    assert result["n_items"] == 1
    assert 0.0 <= result["baseline_accuracy"] <= 1.0
    item = result["per_item"][0]
    assert item["intermediate_band_min_rank"] >= 1
    assert isinstance(item["baseline_correct"], bool)
    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["experiment"] == "two_hop"
