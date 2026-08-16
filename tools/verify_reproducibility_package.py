#!/usr/bin/env python3
"""Release-level consistency checks for the single DT1D-Adapter package."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main()->int:
    required=[
        'models/dt1d_adapter.py','models/dt1d_ablation_adapter.py',
        'tools/run_experiment.py','tools/run_cnn_paper.py','tools/validate_dt1d.py','tools/validate_all_configs.py',
        'configs/experiments/index.yaml','configs/dense/index.yaml',
        'requirements.txt','environment.yml','CITATION.cff','codemeta.json','.zenodo.json','proposal_spec.json','proposal_fingerprint.py','README.md','EXPERIMENTS.md','REPRODUCIBILITY.md','DATASETS.md','KAGGLE_RUN_INSTRUCTIONS.md',
    ]
    missing=[x for x in required if not (ROOT/x).is_file()]
    if missing: raise SystemExit('Missing release files: '+', '.join(missing))
    forbidden=[
        'models/legacy_dt1d_adapter.py','tools/run_experiments_fastest_first.py',
        'models/whc_compact_dt1d_adapter.py','models/whc_final_dt1d_adapter.py','models/whc_tau3_dt1d_adapter.py',
        'configs/proposals','configs/ablations','configs/paper/cnn_three_seed_manifest.yaml',
    ]
    present=[x for x in forbidden if (ROOT/x).exists()]
    if present: raise SystemExit('Obsolete proposal artifacts remain: '+', '.join(present))
    obsolete_docs=[
        'CHANGELOG.md','KAGGLE_CELL.md','KAGGLE_EXPERIMENT_ORDER.md',
        'MANUSCRIPT_ALIGNMENT_NOTES.md','MANUSCRIPT_TO_CODE.md','PRETRAINED_WEIGHTS.md',
        'REVIEWER_RESPONSE_REPRODUCIBILITY.md','REVIEWER_RUNBOOK.md',
        'SINGLE_PROPOSAL_RELEASE_NOTES.md','configs/README.md',
    ]
    stale=[x for x in obsolete_docs if (ROOT/x).exists()]
    if stale: raise SystemExit('Obsolete Markdown files remain: '+', '.join(stale))
    dense_index=yaml.safe_load((ROOT/'configs/dense/index.yaml').read_text())
    assert dense_index['schema_version']==2
    for name in dense_index['experiments']:
        dense_path=ROOT/'configs/dense/experiments'/f'{name}.yaml'
        assert dense_path.is_file(), dense_path
        dense_cfg=yaml.safe_load(dense_path.read_text())
        assert dense_cfg['methods']['dt1d']['args']['tuning_method']=='dt1d'
        assert dense_cfg['fairness']['test_used_for_selection'] is False
        assert dense_cfg['fairness']['evaluate_test_once'] is True
    index=yaml.safe_load((ROOT/'configs/experiments/index.yaml').read_text())
    assert index['schema_version']==4
    assert index['proposal']['method_key']=='dt1d'
    assert index['proposal']['name']=='DT1D-Adapter'
    for name in index['main_experiments']:
        cfg=yaml.safe_load((ROOT/'configs/experiments'/f'{name}.yaml').read_text())
        assert cfg['methods']['dt1d']['args']['tuning_method']=='dt1d'
        assert cfg['fairness']['test_used_for_selection'] is False
        assert cfg['fairness']['evaluate_test_once'] is True
        assert cfg['fairness']['lr_selection_scope'] == 'method_across_seeds'
        assert cfg['fairness']['lr_aggregation'] == 'mean_best_val_acc1'
    version=(ROOT/'VERSION').read_text().strip()
    citation=yaml.safe_load((ROOT/'CITATION.cff').read_text())
    codemeta=json.loads((ROOT/'codemeta.json').read_text())
    zenodo=json.loads((ROOT/'.zenodo.json').read_text())
    assert citation['version']==codemeta['version']==zenodo['version']==version
    assert citation['repository-code']=='https://github.com/tydeptrai21042004/whc-dt1d'
    from proposal_contract import proposal_fingerprint
    assert index['proposal']['architecture']==json.loads((ROOT/'proposal_spec.json').read_text())['architecture']
    fingerprint=proposal_fingerprint(ROOT)
    print(json.dumps({'status':'PASS','proposal':index['proposal'],'main_experiments':len(index['main_experiments']),'reviewer_ablations':len(index['reviewer_ablations']),'dense_experiments':len(dense_index['experiments']),'proposal_fingerprint_sha256':fingerprint},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
