from __future__ import annotations

import main


def make_args(method: str):
    parser = main.get_args_parser()
    args = parser.parse_args([])
    args.backbone = "resnet18"
    args.weights = "none"
    args.nb_classes = 5
    args.tuning_method = method
    return main.canonicalize_args(args)


def build(method: str):
    args = make_args(method)
    model, ids = main.build_model_for_experiment(args)
    main.set_trainability_policy(model, args, ids)
    return model


def test_full_finetuning_trains_every_parameter():
    model = build("full")
    params = list(model.parameters())
    assert params
    assert all(p.requires_grad for p in params)


def test_linear_probe_trains_only_classifier():
    model = build("linear")
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    assert trainable
    assert frozen
    assert all(main._is_head_param(name) for name in trainable)
    assert not any(main._is_head_param(name) for name in frozen)
