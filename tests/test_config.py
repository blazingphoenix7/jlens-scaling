from jlens_scaling.config import load_config


def test_load_config_roundtrip(tmp_path):
    cfg_file = tmp_path / "m.yaml"
    cfg_file.write_text(
        """
model:
  model_id: openai-community/gpt2
  revision: main
  dtype: float32
  slug: gpt2-small
  chat: false
fit:
  n_prompts: 100
  max_seq_len: 128
  dim_batch: 16
  corpus: wikitext-103
  corpus_seed: 0
  min_tokens: 64
""",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    assert cfg.model.slug == "gpt2-small"
    assert cfg.model.chat is False
    assert cfg.fit.n_prompts == 100
    assert cfg.fit.dim_batch == 16
