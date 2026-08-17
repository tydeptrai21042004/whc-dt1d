from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def test_publication_seed_policy_is_fail_closed():
    idx=yaml.safe_load((ROOT/'configs/experiments/index.yaml').read_text())
    for name in idx['table_experiments']+idx['reviewer_ablations']:
        cfg=yaml.safe_load((ROOT/'configs/experiments'/f'{name}.yaml').read_text()); assert cfg['seed_policy']=='multi_seed_table'; assert len(cfg['seeds'])>=3 and len(cfg['seeds'])==len(set(cfg['seeds']))
    for name in idx['figure_experiments']:
        cfg=yaml.safe_load((ROOT/'configs/experiments'/f'{name}.yaml').read_text()); assert cfg['seed_policy']=='single_seed_figure'; assert cfg['seeds']==[0]
