from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "README.md",
    "EXPERIMENTS.md",
    "REPRODUCIBILITY.md",
    "DATASETS.md",
    "KAGGLE_RUN_INSTRUCTIONS.md",
}


def _docs_text() -> str:
    return "\n".join((ROOT / name).read_text(encoding="utf-8") for name in sorted(ALLOWED))


def test_markdown_surface_is_minimal_and_intentional():
    found = {
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*.md")
        if ".pytest_cache" not in path.parts
    }
    assert found == ALLOWED


def test_docs_do_not_expose_removed_proposal_names():
    text = _docs_text().lower()
    for forbidden in ("whc_final", "whc-final", "whc-dt1d", "whc_tau3", "whc_dt"):
        assert forbidden not in text


def test_docs_do_not_contain_obsolete_generated_config_commands():
    text = _docs_text()
    for forbidden in (
        "_generated_configs",
        "scripts/tables/",
        "MANUSCRIPT_TO_CODE.md",
        "PRETRAINED_WEIGHTS.md",
        "KAGGLE_CELL.md",
    ):
        assert forbidden not in text


def test_documented_primary_commands_exist():
    for relative in (
        "tools/run_experiment.py",
        "tools/run_cnn_paper.py",
        "tools/aggregate_cnn_paper.py",
        "tools/verify_reproducibility_package.py",
        "tools/validate_dt1d.py",
        "tools/validate_all_configs.py",
        "scripts/validate_release.sh",
        "KAGGLE_CNN_THREE_SEED_RUN.sh",
    ):
        assert (ROOT / relative).is_file(), relative


def test_readme_states_single_proposal_and_shared_lr_rule():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "exactly one proposal method" in text
    assert "--tuning_method dt1d" in text
    assert "one LR for the method" in text
    assert "test split is never used" in text
