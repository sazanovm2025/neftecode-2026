#!/usr/bin/env python3
"""End-to-end data pipeline smoke test on real data in data/raw/.

Run from repo root:

    python scripts/smoke_test.py

Prints raw numbers, no asserts. Crashes loudly on any unexpected state —
stack traces are intentional, we want to see the first thing that breaks.

Produces ``data/processed/{vocab,resolver,normalizer}.pkl`` along the way
so the pipeline is ready for the training step.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sep(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _positions_to_labels(schema: list[tuple[str, int]]) -> dict[int, str]:
    out: dict[int, str] = {}
    off = 0
    for name, w in schema:
        for j in range(w):
            out[off + j] = f"{name}[{j}]" if w > 1 else name
        off += w
    return out


def _print_resolve(r: dict, feat_props: list[str]) -> None:
    for p in feat_props:
        v = r[p]
        mflag = int(r[p + "__is_measured"])
        vs = f"{v:.4g}" if not (isinstance(v, float) and np.isnan(v)) else "NaN"
        # truncate long Russian names for readability
        name = p if len(p) <= 60 else p[:57] + "..."
        print(f"    {name:<60s}  -> {vs:>12s}   measured={mflag}")


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------


def step_1_loader():
    sep("1. LOADER — load_raw() on data/raw/")
    from src.data import load_raw

    train, test, props = load_raw(ROOT / "data" / "raw")
    print(f"train: shape={train.shape}")
    print(f"test:  shape={test.shape}")
    print(f"props: shape={props.shape}")
    print("\ntrain head(2):")
    print(train.head(2).to_string())
    print("\ntest head(2):")
    print(test.head(2).to_string())
    print("\nprops head(2):")
    print(props.head(2).to_string())
    return train, test, props


def step_2_vocab(train, test):
    sep("2. VOCAB — build + roundtrip")
    from src.data import UNK, Vocab

    vocab = Vocab().build(train, test)
    print(f"n_components={vocab.n_components}  n_types={vocab.n_types}  n_modes={vocab.n_modes}")
    print(f"UNK in component_id_to_idx?    {UNK in vocab.component_id_to_idx}  (idx={vocab.component_id_to_idx.get(UNK)})")
    print(f"UNK in component_type_to_idx?  {UNK in vocab.component_type_to_idx}  (idx={vocab.component_type_to_idx.get(UNK)})")

    print("\nall component types (idx, label):")
    for t, i in vocab.component_type_to_idx.items():
        print(f"  [{i:2d}]  {t}")

    print(f"\nall {vocab.n_modes} modes (idx, T °C, t h, biofuel %, catalyst cat):")
    for mode, i in vocab.mode_id_to_idx.items():
        T, t_h, bf, cat = mode
        print(f"  [{i:2d}]  T={T:>5.1f}  t={t_h:>5.1f}  biofuel={bf:>4.1f}  cat={cat}")

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    vocab.save(out / "vocab.pkl")
    vocab2 = Vocab.load(out / "vocab.pkl")
    ok = (
        vocab.component_id_to_idx == vocab2.component_id_to_idx
        and vocab.component_type_to_idx == vocab2.component_type_to_idx
        and vocab.mode_id_to_idx == vocab2.mode_id_to_idx
        and vocab.T_cats == vocab2.T_cats
        and vocab.t_cats == vocab2.t_cats
        and vocab.biofuel_cats == vocab2.biofuel_cats
        and vocab.catalyst_cats == vocab2.catalyst_cats
    )
    print(f"\npickle roundtrip: {'OK' if ok else 'MISMATCH !!!'}")
    return vocab


def step_3_properties(train, test, props, vocab):
    sep("3. PROPERTIES — wide matrix, coverage, 3-level resolve")
    from src.data import PropertyResolver

    resolver = PropertyResolver.build(props, train, test, vocab, k=15)
    print(f"wide matrix shape (pairs × properties): {resolver.wide.shape}")
    print(f"kv100 col: {resolver.kv100_col!r}")
    print(f"tbn col:   {resolver.tbn_col!r}")

    # recompute coverage (measured ∨ typical fallback) for display
    mix_pairs = (
        pd.concat(
            [train[["component_id", "batch_id"]], test[["component_id", "batch_id"]]]
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )
    pairs_idx = pd.MultiIndex.from_frame(mix_pairs, names=["component_id", "batch_id"])
    measured = resolver.wide.reindex(pairs_idx)
    typ = resolver.wide[resolver.wide.index.get_level_values(1) == "typical"].copy()
    typ.index = typ.index.get_level_values(0)
    typ_for_pairs = typ.reindex(mix_pairs["component_id"].to_numpy())
    typ_for_pairs.index = pairs_idx
    combined = measured.fillna(typ_for_pairs)

    print(f"\ntop-15 feature_properties (coverage over {len(mix_pairs)} mix pairs):")
    for p in resolver.feature_properties:
        cov = (
            float(combined[p].notna().mean()) if p in combined.columns else float("nan")
        )
        name = p if len(p) <= 60 else p[:57] + "..."
        print(f"  [{cov:5.1%}]  {name}")

    # CASE A — a real (component, batch) in train with at least one measured top-15 prop
    case_a = None
    for row in train[["component_id", "batch_id"]].drop_duplicates().itertuples(index=False):
        if row.batch_id == "typical":
            continue
        if (row.component_id, row.batch_id) not in resolver.wide.index:
            continue
        r = resolver.resolve(row.component_id, row.batch_id)
        if any(r[p + "__is_measured"] == 1.0 for p in resolver.feature_properties):
            case_a = (row.component_id, row.batch_id)
            break
    print(f"\n--- CASE A (level 1: measured) --- pair={case_a}")
    if case_a is not None:
        _print_resolve(resolver.resolve(*case_a), resolver.feature_properties)

    # CASE B — mix pair whose exact (comp,batch) is absent but (comp,'typical') exists
    typical_comps = set(
        resolver.wide[resolver.wide.index.get_level_values(1) == "typical"]
        .index.get_level_values(0)
    )
    case_b = None
    for row in (
        pd.concat([train, test])[["component_id", "batch_id"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        if row.batch_id == "typical":
            continue
        if (row.component_id, row.batch_id) in resolver.wide.index:
            continue
        if row.component_id in typical_comps:
            case_b = (row.component_id, row.batch_id)
            break
    print(f"\n--- CASE B (level 2: typical fallback) --- pair={case_b}")
    if case_b is not None:
        _print_resolve(resolver.resolve(*case_b), resolver.feature_properties)

    # CASE C — pair with neither measured nor typical
    case_c = None
    for row in (
        pd.concat([train, test])[["component_id", "batch_id"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        if row.batch_id == "typical":
            continue
        if row.component_id in typical_comps:
            continue
        if (row.component_id, row.batch_id) in resolver.wide.index:
            continue
        case_c = (row.component_id, row.batch_id)
        break
    print(f"\n--- CASE C (level 3: NaN, then impute class median) --- pair={case_c}")
    if case_c is not None:
        r_raw = resolver.resolve(*case_c)
        print("  before impute:")
        _print_resolve(r_raw, resolver.feature_properties)
        resolver.impute_class_median(case_c[0], r_raw)
        print("  after impute_class_median:")
        _print_resolve(r_raw, resolver.feature_properties)
    else:
        print("  (no such pair exists in train+test — all components have at least typical data)")

    resolver.save(ROOT / "data" / "processed" / "resolver.pkl")
    return resolver


def step_4_features_train1(train, resolver, vocab):
    sep("4. FEATURES — train_1 in detail")
    from src.data import (
        build_component_features,
        build_scenario_features,
        component_feature_schema,
        scenario_feature_schema,
    )

    scen = train[train["scenario_id"] == "train_1"]
    print(f"raw train_1 rows ({len(scen)} components):")
    print(scen[["component_id", "batch_id", "share_pct"]].to_string(index=False))

    cf = build_component_features(scen, resolver, vocab)
    print(f"\ncomponent_features.shape = {cf.shape}")
    print("schema:")
    offset = 0
    for name, width in component_feature_schema(resolver):
        print(f"  cols [{offset:2d}:{offset + width:2d}]  {name}  (width={width})")
        offset += width
    np.set_printoptions(precision=3, suppress=True, linewidth=180)
    print("\nfirst 2 rows of component_features:")
    for i in range(min(2, cf.shape[0])):
        print(f"  row[{i}] = {cf[i]}")

    sf = build_scenario_features(scen, resolver, vocab)
    print(f"\nscenario_features.shape = {sf.shape}")
    print("decoded scenario features:")
    offset = 0
    type_list = list(vocab.component_type_to_idx.keys())
    idx_to_mode = {v: k for k, v in vocab.mode_id_to_idx.items()}
    for name, width in scenario_feature_schema(vocab, resolver):
        chunk = sf[offset : offset + width]
        if name == "mode_id_idx":
            m = idx_to_mode[int(chunk[0])]
            label = f"mode_id={int(chunk[0])}  (T={m[0]}, t={m[1]}, bf={m[2]}, cat={m[3]})"
        elif name == "T_onehot":
            label = f"T_onehot={list(chunk)}  cats={vocab.T_cats}"
        elif name == "t_onehot":
            label = f"t_onehot={list(chunk)}  cats={vocab.t_cats}"
        elif name == "biofuel_onehot":
            label = f"biofuel_onehot={list(chunk)}  cats={vocab.biofuel_cats}"
        elif name == "catalyst_onehot":
            label = f"catalyst_onehot={list(chunk)}  cats={vocab.catalyst_cats}"
        elif name == "has_class":
            on = [type_list[i] for i, v in enumerate(chunk) if v > 0.5]
            label = f"has_class: on={on}"
        elif name == "mean_share_class":
            vals = {
                type_list[i]: round(float(chunk[i]), 3)
                for i in range(len(chunk))
                if chunk[i] > 0
            }
            label = f"mean_share_class: {vals}"
        elif width == 1:
            label = f"{name} = {float(chunk[0]):.6g}"
        else:
            label = f"{name} = {[round(float(x), 4) for x in chunk]}"
        print(f"  [{offset:2d}:{offset + width:2d}]  {label}")
        offset += width
    return cf, sf


def step_5_build_all(train, test, resolver, vocab):
    sep("5. BUILD ALL records + sanity checks")
    from src.data import build_scenario_record

    train_records = [
        build_scenario_record(sid, grp, resolver, vocab, has_targets=True)
        for sid, grp in train.groupby("scenario_id", sort=False)
    ]
    test_records = [
        build_scenario_record(sid, grp, resolver, vocab, has_targets=False)
        for sid, grp in test.groupby("scenario_id", sort=False)
    ]

    train_ns = [r["component_features"].shape[0] for r in train_records]
    test_ns = [r["component_features"].shape[0] for r in test_records]
    D_comp = train_records[0]["component_features"].shape[1]
    D_scen = train_records[0]["scenario_features"].shape[0]

    print(f"train: {len(train_records)} records, N_comp min={min(train_ns)}  median={int(np.median(train_ns))}  max={max(train_ns)}")
    print(f"test:  {len(test_records)} records, N_comp min={min(test_ns)}  median={int(np.median(test_ns))}  max={max(test_ns)}")
    print(f"D_comp={D_comp}  D_scen={D_scen}")
    print(f"train padded tensor would be: ({len(train_records)}, {max(train_ns)}, {D_comp})")
    print(f"test padded tensor would be:  ({len(test_records)},  {max(test_ns)}, {D_comp})")

    # sanity A — batch_known distribution on train (col idx 4)
    total_train_comps = int(sum(train_ns))
    known_train = sum(int(r["component_features"][:, 4].sum()) for r in train_records)
    print(
        f"\nsanity (train) batch_known=1: {known_train}/{total_train_comps} "
        f"({known_train / total_train_comps:.1%}) — expect 100% for train"
    )

    # sanity B — mode_id distribution (feature vector position 0)
    print("\nsanity (train) mode_id distribution:")
    idx_to_mode = {v: k for k, v in vocab.mode_id_to_idx.items()}
    mode_counts = Counter(int(r["scenario_features"][0]) for r in train_records)
    for idx in sorted(mode_counts):
        T, t_h, bf, cat = idx_to_mode[idx]
        print(f"  mode_id={idx:2d}  T={T:>5.1f} t={t_h:>5.1f} bf={bf:>4.1f} cat={cat}  ->  {mode_counts[idx]:3d} scen")

    # sanity C — NaN in any features after imputation
    def has_nan(rec):
        return (
            bool(np.isnan(rec["component_features"]).any())
            or bool(np.isnan(rec["scenario_features"]).any())
        )

    nan_train = sum(1 for r in train_records if has_nan(r))
    print(f"\nsanity (train) records with NaN in features = {nan_train}  (expect 0)")

    # test sanity — batch_known
    total_test_comps = int(sum(test_ns))
    known_test = sum(int(r["component_features"][:, 4].sum()) for r in test_records)
    unknown_test = total_test_comps - known_test
    print(
        f"\nsanity (test) batch_known=1: {known_test}/{total_test_comps} "
        f"({known_test / total_test_comps:.1%})"
    )
    print(f"  -> {unknown_test} test components with batch_known=0")

    return train_records, test_records


def step_6_normalizer(train_records, resolver, vocab):
    sep("6. NORMALIZER — fit on train, stats on first 5 numeric comp cols")
    from src.data import (
        FeatureNormalizer,
        component_feature_schema,
        component_numeric_mask,
        scenario_numeric_mask,
    )

    cnum = component_numeric_mask(resolver)
    snum = scenario_numeric_mask(vocab, resolver)
    print(f"component numeric columns: {int(cnum.sum())}/{len(cnum)}")
    print(f"scenario numeric columns:  {int(snum.sum())}/{len(snum)}")

    norm = FeatureNormalizer().fit(train_records, cnum, snum)

    labels = _positions_to_labels(component_feature_schema(resolver))
    numeric_positions = np.where(cnum)[0][:5]
    print("\nfirst 5 numeric component columns — train stats:")
    for pos in numeric_positions:
        print(
            f"  col[{pos:2d}]  mean={norm.comp_mean[pos]:+13.4f}  std={norm.comp_std[pos]:13.4f}  ({labels[pos]})"
        )

    _ = norm.transform_record(train_records[0])
    print("\napply to train_records[0] via transform_record(): OK")

    norm.save(ROOT / "data" / "processed" / "normalizer.pkl")
    return norm


def step_7_target_transformer(train_records):
    sep("7. TARGET TRANSFORMER — fit + roundtrip on train_1")
    from src.utils.transforms import TargetTransformer

    df = pd.DataFrame(
        {
            "target_dkv": [float(r["targets"][0]) for r in train_records],
            "target_eot": [float(r["targets"][1]) for r in train_records],
        }
    )
    tt = TargetTransformer().fit(df)
    print(f"raw stats:          {tt.raw_stats()}")
    print(f"transformed means:  {tt.mean_}")
    print(f"transformed stds:   {tt.std_}")

    t1 = df.iloc[:1]
    raw_dkv = float(t1["target_dkv"].iloc[0])
    raw_eot = float(t1["target_eot"].iloc[0])
    print(f"\ntrain_1 raw targets: dkv={raw_dkv:.6f}, eot={raw_eot:.6f}")

    z = tt.transform(t1)
    print(f"train_1 z-space:     dkv={float(z['target_dkv'][0]):+.6f}, eot={float(z['target_eot'][0]):+.6f}")

    back = tt.inverse_transform(z)
    back_dkv = float(back["target_dkv"][0])
    back_eot = float(back["target_eot"][0])
    print(f"train_1 inversed:    dkv={back_dkv:.6f}, eot={back_eot:.6f}")

    drift_dkv = abs(back_dkv - raw_dkv)
    drift_eot = abs(back_eot - raw_eot)
    print(f"roundtrip drift:     |dkv|={drift_dkv:.2e}, |eot|={drift_eot:.2e}   (expect < 1e-6)")
    return tt


def step_8_dataset_dataloader(train_records, normalizer, target_transformer):
    sep("8. DATASET + COLLATE_FN — batch of 4")
    import torch
    from torch.utils.data import DataLoader

    from src.data import DaimlerDataset, collate_fn

    ds = DaimlerDataset(
        train_records[:4],
        normalizer=normalizer,
        target_transformer=target_transformer,
        training=True,
        component_dropout_p=0.0,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    print("batch keys and shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:<22s}: shape={tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:<22s}: {v}")


def step_9_imputation_coverage(train, resolver):
    sep("9. IMPUTATION COVERAGE DIAGNOSTIC")
    from collections import defaultdict

    from src.data.vocab import extract_component_type

    feat_props = resolver.feature_properties

    # For each (component_id, batch_id) pair that appears in a train recipe:
    # - resolve with measured → typical (level 1+2), record whether we got a value
    # - then apply impute_class_median and check if any NaN remains (level 3 fails)
    train_pairs = train[["component_id", "batch_id"]].drop_duplicates()

    # class → property → counts
    counts = defaultdict(lambda: defaultdict(lambda: {"total": 0, "covered": 0, "nan_after_impute": 0}))
    for row in train_pairs.itertuples(index=False):
        cid = str(row.component_id)
        bid = str(row.batch_id)
        klass = extract_component_type(cid)

        before = resolver.resolve(cid, bid)
        after = {k: v for k, v in before.items()}
        resolver.impute_class_median(cid, after)

        for p in feat_props:
            s = counts[klass][p]
            s["total"] += 1
            if not np.isnan(before[p]):
                s["covered"] += 1
            if np.isnan(after[p]):
                s["nan_after_impute"] += 1

    classes = sorted(counts.keys())

    # 1) completely-missing (class, property) pairs
    missing = []
    for klass in classes:
        for p in feat_props:
            s = counts[klass][p]
            if s["covered"] == 0 and s["total"] > 0:
                missing.append((klass, p, s["total"]))
    print(f"\n(class, property) pairs with 0 measured/typical values ({len(missing)} total):")
    if missing:
        for klass, p, n in missing:
            name = p if len(p) <= 55 else p[:52] + "..."
            print(f"  class={klass:<30s}  {name:<55s}  (n_comp_in_train={n})")
    else:
        print("  (none)")

    # 2) per-property coverage summary
    print("\nper-property coverage across all train pairs (level 1+2, before class-median):")
    for p in feat_props:
        total = sum(counts[k][p]["total"] for k in classes)
        covered = sum(counts[k][p]["covered"] for k in classes)
        cov = covered / max(1, total)
        name = p if len(p) <= 55 else p[:52] + "..."
        print(f"  [{cov:5.1%}]  {name}")

    # 3) full class × property coverage matrix
    print("\nfull class × property coverage matrix (level 1+2, before class-median):")
    mat = pd.DataFrame(index=classes, columns=feat_props, dtype=float)
    for klass in classes:
        for p in feat_props:
            s = counts[klass][p]
            mat.at[klass, p] = s["covered"] / s["total"] if s["total"] > 0 else np.nan
    # short property labels for display
    short_cols = [c[:22] + ("…" if len(c) > 22 else "") for c in mat.columns]
    mat_disp = mat.copy()
    mat_disp.columns = short_cols
    with pd.option_context(
        "display.max_columns", 50,
        "display.width", 260,
        "display.float_format", lambda v: f"{v*100:5.1f}%" if not pd.isna(v) else "  —  ",
    ):
        print(mat_disp.to_string())

    # 3b) class × is_applicable summary
    print("\nclass_applicability (count of applicable top-15 props per class):")
    for klass in sorted(resolver.class_applicability.keys()):
        d = resolver.class_applicability[klass]
        on = [p for p, v in d.items() if v]
        print(f"  {klass:<30s}  {len(on):>2d}/15  applicable")
    print("\nfull class × is_applicable matrix:")
    applic_mat = pd.DataFrame(
        index=sorted(resolver.class_applicability.keys()),
        columns=feat_props,
        dtype=int,
    )
    for klass in applic_mat.index:
        d = resolver.class_applicability[klass]
        for p in feat_props:
            applic_mat.at[klass, p] = 1 if d.get(p, False) else 0
    applic_disp = applic_mat.copy()
    applic_disp.columns = [c[:22] + ("…" if len(c) > 22 else "") for c in applic_mat.columns]
    with pd.option_context("display.max_columns", 50, "display.width", 260):
        print(applic_disp.to_string())

    # 4) top-3 problematic (class, property) pairs by NaN rate AFTER impute
    problematic = []
    for klass in classes:
        for p in feat_props:
            s = counts[klass][p]
            if s["total"] > 0 and s["nan_after_impute"] > 0:
                problematic.append(
                    (s["nan_after_impute"] / s["total"], klass, p, s["total"], s["nan_after_impute"])
                )
    problematic.sort(reverse=True)
    print("\ntop-3 problematic (class, property) pairs by NaN rate AFTER class-median impute:")
    if not problematic:
        print("  (none — class-median fills every residual NaN)")
    else:
        for rate, klass, p, n, n_nan in problematic[:3]:
            name = p if len(p) <= 55 else p[:52] + "..."
            print(f"  [{rate:5.1%}]  class={klass:<30s}  {name:<55s}  ({n_nan}/{n} still NaN)")


def step_10_quick_checks(train_records, test_records, resolver):
    sep("10. QUICK CHECKS — batch_known + sign_target")
    from src.data import component_feature_schema

    # find batch_known_flag column index from the schema
    schema = component_feature_schema(resolver)
    batch_known_col = None
    off = 0
    for name, w in schema:
        if name == "batch_known":
            batch_known_col = off
            break
        off += w
    print(f"batch_known_col index (from schema) = {batch_known_col}")

    # 1) train batch_known — expect 100%
    train_known = 0
    train_total = 0
    for r in train_records:
        cf = r["component_features"]
        train_known += int(cf[:, batch_known_col].sum())
        train_total += cf.shape[0]
    print(
        f"train batch_known: {train_known}/{train_total} "
        f"({train_known / train_total * 100:.1f}%)"
    )

    # 2) test batch_known=0 — how many unknown?
    test_unknown = 0
    test_total = 0
    for r in test_records:
        cf = r["component_features"]
        test_unknown += int((cf[:, batch_known_col] == 0).sum())
        test_total += cf.shape[0]
    print(f"test unknown batches: {test_unknown}/{test_total}")

    # 3) sign_target distribution on train
    pos = sum(1 for r in train_records if r.get("sign_target") == 1)
    neg = sum(1 for r in train_records if r.get("sign_target") == 0)
    print(f"train sign: {pos} positive, {neg} negative  (expect 92/75)")


def main():
    train, test, props = step_1_loader()
    vocab = step_2_vocab(train, test)
    resolver = step_3_properties(train, test, props, vocab)
    _cf, _sf = step_4_features_train1(train, resolver, vocab)
    train_records, test_records = step_5_build_all(train, test, resolver, vocab)
    normalizer = step_6_normalizer(train_records, resolver, vocab)
    tt = step_7_target_transformer(train_records)
    step_8_dataset_dataloader(train_records, normalizer, tt)
    step_9_imputation_coverage(train, resolver)
    step_10_quick_checks(train_records, test_records, resolver)

    sep("SUMMARY")
    print(f"train records: {len(train_records)}   test records: {len(test_records)}")
    print(f"artifacts written to data/processed/: vocab.pkl, resolver.pkl, normalizer.pkl")
    print("all 8 steps completed without exceptions.")


if __name__ == "__main__":
    main()
