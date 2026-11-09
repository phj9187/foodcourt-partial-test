"""Partial / confirmation runs on top of a finished foodcourt-adaptive-sim experiment.

Re-trains chosen objectives (existing ids such as lhs_0022, or custom weights)
with NEW replica indices, using the exact config and source of the main run,
then pools them with the main run's replicas and reports customer / profit / F.

Same job code as the main run (kiosk_adaptive.study.run_job): a job's seed is
seed_for(config seed, objective id, replica), so new replica numbers give
independent seeds while everything else (courts, worlds, menus) stays identical.
"""
import argparse, json, math, multiprocessing, os, sys, time
# Colab TPU runtimes ship torch_tpu; when it auto-loads, torch.manual_seed tries to seed the TPU and
# parallel workers collide on /tmp/libtpu_lockfile. The simulation is CPU-only, so never load it.
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
os.environ.setdefault('PJRT_DEVICE', 'CPU')
from dataclasses import replace, asdict
from datetime import datetime, timezone
from pathlib import Path


def parse_reps(text):
    reps = []
    for part in text.split(','):
        if '-' in part:
            a, b = part.split('-'); reps += list(range(int(a), int(b)+1))
        elif part:
            reps.append(int(part))
    if not reps or min(reps) < 0 or len(set(reps)) != len(reps):
        raise ValueError('bad --reps')
    return reps


def parse_custom(text):
    # name:a,b,c,lambda,q   (a,b,c = taste, affordability, waiting weights; normalised to sum 1)
    name, values = text.split(':')
    a, b, c, lam, q = map(float, values.split(','))
    if not all(math.isfinite(v) and v >= 0 for v in (a, b, c)) or a+b+c <= 0:
        raise ValueError('custom a,b,c must be nonnegative, not all zero')
    if not (0 <= lam <= 1 and 0 <= q <= 1):
        raise ValueError('custom lambda and q must be in [0,1]')
    if not name.replace('_', '').isalnum():
        raise ValueError('custom name: letters, digits, underscore')
    s = a+b+c
    return 'custom_'+name, (a/s, b/s, c/s, lam, q)


RANGE_KEYS = ('lambda', 'q', 'b', 'a_share')


def parse_range(text):
    # "lambda=0.65..0.95,q=0.35..0.65,b=0..0.1,a_share=0..1"
    # b = affordability weight; a_share = taste/(taste+waiting) for the rest 1-b.
    box = dict(lambda_=(0., 1.), q=(0., 1.), b=(0., 1.), a_share=(0., 1.))
    box = {k.rstrip('_'): v for k, v in box.items()}
    for part in filter(None, (x.strip() for x in text.split(','))):
        key, span = part.split('=')
        lo, hi = map(float, span.split('..'))
        if key not in RANGE_KEYS or not 0 <= lo <= hi <= 1:
            raise ValueError(f'bad range {part}; keys {RANGE_KEYS}, values in [0,1]')
        box[key] = (lo, hi)
    return box


def sample_box(n, box, seed, prefix='box'):
    """Latin hypercube inside the box; returns custom objectives (name, (a,b,c,lambda,q))."""
    import numpy as np
    rng = np.random.default_rng(seed)
    cube = np.column_stack([(rng.permutation(n)+rng.random(n))/n for _ in RANGE_KEYS])
    out = []
    for i, x in enumerate(cube):
        v = {k: box[k][0]+(box[k][1]-box[k][0])*float(x[j]) for j, k in enumerate(RANGE_KEYS)}
        b = v['b']; a = (1-b)*v['a_share']; c = 1-b-a
        out.append((f'custom_{prefix}_{i:02d}', (a, b, c, v['lambda'], v['q'])))
    return out


def load_config(source, config, overrides=()):
    import yaml
    from kiosk_adaptive.study import DEFAULT
    cfg = dict(DEFAULT, **yaml.safe_load((Path(source)/config).read_text(encoding='utf-8')))
    for item in overrides:
        key, value = item.split('=', 1)
        if key not in DEFAULT:
            raise ValueError(f'unknown setting {key} for this source version')
        cfg[key] = yaml.safe_load(value)
    cfg['audit_thresholds'] = dict(DEFAULT['audit_thresholds'], **(cfg['audit_thresholds'] or {}))
    return cfg


def build(cfg):
    from kiosk_adaptive.study import make_court
    from kiosk_adaptive.models import Settings
    settings = Settings(seed=cfg['seed'], recommendation_count=1, venue_types=tuple(cfg['venue_types']),
                        venue_lhs_candidates=cfg['venue_lhs_candidates'], taste_noise_override=cfg['taste_noise_override'],
                        input_noise=cfg['input_noise'], input_missing_probability=cfg['input_missing_probability'],
                        stock_basis=cfg['stock_basis'],
                        **({'input_minutes': cfg['input_minutes']} if 'input_minutes' in cfg else {})).validate()
    catalog = {s: [make_court(settings, i, s, cfg['visitors'], cfg[s+'_courts']) for i in range(cfg[s+'_courts'])]
               for s in ('train', 'validation', 'test')}
    return settings, catalog


def select_objectives(cfg, ids, customs):
    from kiosk_adaptive.design import lhs_objectives, Objective
    power = cfg['powers'][0]
    pool, _ = lhs_objectives(cfg['lhs_samples'], cfg['seed'])
    pool = {o.id: replace(o, id=o.id+f'_p{power:g}', power=power) for o in pool}
    chosen = []
    for i in ids:
        key = i[:-len(f'_p{power:g}')] if i.endswith(f'_p{power:g}') else i
        if key not in pool:
            raise ValueError(f'unknown objective {i}; available: {sorted(pool)}')
        chosen.append(pool[key])
    for name, w in customs:
        chosen.append(Objective(name+f'_p{power:g}', *w, power=power))
    # a custom id must not reuse a main-run id (seeds come from the id)
    if any(o.id in pool or o.id[:-len(f'_p{power:g}')] in pool for o in chosen[len(ids):]):
        raise ValueError('custom name collides with a main-run objective id')
    if len({o.id for o in chosen}) != len(chosen):
        raise ValueError('duplicate objective')
    return chosen


_STATE = {}


def _init(source, config, output, overrides=()):
    sys.path.insert(0, str(Path(source)/'src'))
    cfg = load_config(source, config, overrides)
    settings, catalog = build(cfg)
    audit_path = Path(output)/'baseline_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else None
    _STATE.update(cfg=cfg, settings=settings, catalog=catalog, audit=audit, output=Path(output))


def _job(objective_dict, rep):
    from kiosk_adaptive.design import Objective
    from kiosk_adaptive.study import run_job
    s = _STATE
    return run_job(Objective(**objective_dict), rep, s['cfg'], s['settings'], s['catalog'], s['audit'], s['output'])


def t95(values):
    from kiosk_adaptive.study import t975
    n = len(values)
    if n < 2:
        return None
    m = sum(values)/n
    sd = (sum((v-m)**2 for v in values)/(n-1))**.5
    h = t975(n-1)*sd/n**.5
    return [m-h, m+h]


def read_summaries(folder, objective_id):
    out = {}
    root = Path(folder)/objective_id if folder else None
    if root and root.exists():
        for p in sorted(root.glob('rep_*/summary.json')):
            s = json.loads(p.read_text(encoding='utf-8'))
            out[s['replica']] = s
    return out


def aggregate(objectives, output, reference, seed, sensitivity=False):
    from kiosk_adaptive.study import combine_replicas
    report, rows = {}, []
    for o in objectives:
        new = read_summaries(output, o.id)
        old = read_summaries(reference, o.id)
        if sensitivity:
            # Changed environment: never pool. Compare with the main run's same replicas (same seeds).
            old = {r: v for r, v in old.items() if r in new}
            both = {}
        else:
            both = {**old, **new}
        entry = dict(objective=asdict(o), reference_replicas=sorted(old), new_replicas=sorted(new))
        for label, group in (('new_only', new), ('reference_same_reps' if sensitivity else 'reference_only', old), ('pooled', both)):
            if not group:
                continue
            sums = [group[r] for r in sorted(group)]
            part = {}
            for scenario in ('stable', 'changed'):
                res = [s['results'][scenario] for s in sums if scenario in s['results']]
                if not res:
                    continue
                cust = [r['mean_customer_delta'] for r in res]
                prof = [r['mean_profit_delta_won'] for r in res]
                f = [r['mean_delta'] for r in res]
                lc = [r['learning_check']['ai_minus_best_rule']['mean'] for r in res if 'learning_check' in r]
                pooled = combine_replicas(sums, seed)[o.id].get(scenario, {})
                part[scenario] = dict(replicas=len(res),
                    customer_delta_per100=100*sum(cust)/len(cust), customer_t95_per100=[100*x for x in t95(cust)] if t95(cust) else None,
                    profit_delta_won=sum(prof)/len(prof), profit_t95_won=t95(prof),
                    profit_replica_min=min(prof), profit_replica_max=max(prof),
                    F_delta=sum(f)/len(f), F_court_t95=pooled.get('t95'), F_harm_fraction=pooled.get('harm_fraction'),
                    ai_minus_best_rule=sum(lc)/len(lc) if lc else None)
                rows.append(dict(id=o.id, group=label, scenario=scenario, **{k: v for k, v in part[scenario].items() if not isinstance(v, list)},
                                 customer_lo=(part[scenario]['customer_t95_per100'] or [None])[0],
                                 customer_hi=(part[scenario]['customer_t95_per100'] or [None, None])[1],
                                 profit_lo=(part[scenario]['profit_t95_won'] or [None])[0],
                                 profit_hi=(part[scenario]['profit_t95_won'] or [None, None])[1]))
            entry[label] = part
        if sensitivity and old:
            paired = {}
            for scenario in ('stable', 'changed'):
                reps_ = [r for r in sorted(new) if r in old and scenario in new[r]['results'] and scenario in old[r]['results']]
                if reps_:
                    d = lambda k, m=1.: [m*(new[r]['results'][scenario][k]-old[r]['results'][scenario][k]) for r in reps_]
                    paired[scenario] = dict(replicas=reps_,
                        customer_change_per100=d('mean_customer_delta', 100.), profit_change_won=d('mean_profit_delta_won'),
                        F_change=d('mean_delta'))
            entry['change_vs_reference'] = paired
        report[o.id] = entry
    return report, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, help='checked-out foodcourt-adaptive-sim folder')
    ap.add_argument('--config', default='config/adaptive_extended.yaml')
    ap.add_argument('--output', required=True, help='new experiment folder for these runs')
    ap.add_argument('--reference', default=None, help='finished main experiment folder to pool with (read only)')
    ap.add_argument('--objectives', default='', help='comma list of existing ids, e.g. lhs_0022,lhs_0019')
    ap.add_argument('--custom', action='append', default=[], help='name:a,b,c,lambda,q (repeatable)')
    ap.add_argument('--sample', type=int, default=0, help='number of weightings to draw inside --range (LHS)')
    ap.add_argument('--range', default='', help='lambda=lo..hi,q=lo..hi,b=lo..hi,a_share=lo..hi')
    ap.add_argument('--sample-seed', type=int, default=20261109)
    ap.add_argument('--reps', default='5-8', help='NEW replica indices, e.g. 5-8 or 5,6,7')
    ap.add_argument('--workers', type=int, default=multiprocessing.cpu_count())
    ap.add_argument('--set', action='append', default=[], help='override a config value, e.g. input_minutes=0.5 (sensitivity run: compared with the reference, never pooled)')
    ap.add_argument('--dry-run', action='store_true', help='only print the chosen weightings')
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(args.source)/'src'))
    cfg = load_config(args.source, args.config, args.set)
    sensitivity = bool(args.set)
    ids = [x.strip() for x in args.objectives.split(',') if x.strip()]
    customs = [parse_custom(c) for c in args.custom]
    if args.sample:
        customs += sample_box(args.sample, parse_range(args.range), args.sample_seed)
    objectives = select_objectives(cfg, ids, customs)
    if not objectives:
        raise SystemExit('choose --objectives and/or --custom')
    reps = parse_reps(args.reps)
    if sensitivity:
        print('sensitivity settings:', ', '.join(args.set), flush=True)
    print('objectives (a=taste, b=price, c=waiting, lambda=customer share, q=profit vs waste):', flush=True)
    for o in objectives:
        print(f'  {o.id}: a={o.taste:.3f} b={o.affordability:.3f} c={o.waiting:.3f} lambda={o.customer_share:.3f} q={o.profit_share:.3f}', flush=True)
    if args.dry_run:
        return
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    reference = Path(args.reference) if args.reference else None
    if reference and not (reference/'manifest.json').exists():
        raise SystemExit(f'reference folder has no manifest.json: {reference}')
    if reference and not sensitivity:
        clash = [f'{o.id}/rep_{r}' for o in objectives for r in reps if (reference/o.id/f'rep_{r}'/'summary.json').exists()]
        if clash:
            raise SystemExit('these replicas already exist in the reference run (same seed, not independent): '+', '.join(clash))

    import torch, numpy
    plan = dict(schema=1, overrides=args.set, source=str(args.source), config=args.config, config_values={k: v for k, v in cfg.items() if k != 'workers'},
                objectives=[asdict(o) for o in objectives], reps=reps, reference=str(reference) if reference else None,
                versions=dict(torch=torch.__version__, numpy=numpy.__version__))
    plan_path = output/'partial_plan.json'
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding='utf-8'))
        if {k: old.get(k) for k in ('config_values', 'objectives', 'reps')} != {k: plan[k] for k in ('config_values', 'objectives', 'reps')}:
            raise SystemExit('this folder holds a different partial test; use a new experiment id')
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding='utf-8')
    # baseline audit only labels flagged test courts; reuse the main run's copy when available
    if reference and (reference/'baseline_audit.json').exists() and not (output/'baseline_audit.json').exists():
        (output/'baseline_audit.json').write_bytes((reference/'baseline_audit.json').read_bytes())

    def write_report():
        report, rows = aggregate(objectives, output, reference, cfg['seed'], sensitivity)
        (output/'partial_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
        import csv
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with (output/'partial_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
                w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
        return report

    if not args.aggregate_only:
        jobs = [(o, r) for r in reps for o in objectives]
        print(f'jobs={len(jobs)} workers={args.workers} objectives={[o.id for o in objectives]} reps={reps}', flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)), mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_init, initargs=(str(args.source), args.config, str(output), tuple(args.set))) as pool:
            futures = {pool.submit(_job, asdict(o), r): (o.id, r) for o, r in jobs}
            failed = []
            for fut in as_completed(futures):
                oid, r = futures[fut]
                try:
                    s = fut.result()['results']['stable']
                except Exception as exc:  # one bad job must not throw away the others; rerun resumes it
                    failed.append(f'{oid} rep={r}')
                    print(f'FAILED {oid} rep={r}: {exc!r}', flush=True)
                    continue
                print(f'DONE {oid} rep={r} F={s["mean_delta"]:+.5f} customer={100*s["mean_customer_delta"]:+.3f}/100 '
                      f'profit={s["mean_profit_delta_won"]:+.0f}won elapsed={(time.time()-t0)/60:.1f}min', flush=True)
                write_report()
        if failed:
            print('FAILED jobs (rerun this cell to retry them): '+', '.join(failed), flush=True)
    report = write_report()
    for oid, e in report.items():
        for label in ('new_only', 'reference_same_reps', 'pooled'):
            s = e.get(label, {}).get('stable')
            if s:
                print(f'{oid} [{label}, n={s["replicas"]}] customer {s["customer_delta_per100"]:+.3f}/100 '
                      f'profit {s["profit_delta_won"]:+.0f} won  F {s["F_delta"]:+.5f}', flush=True)
        ch = e.get('change_vs_reference', {}).get('stable')
        if ch:
            m = lambda v: sum(v)/len(v)
            print(f'{oid} [change vs main run, same seeds, reps {ch["replicas"]}] customer {m(ch["customer_change_per100"]):+.3f}/100 '
                  f'profit {m(ch["profit_change_won"]):+.0f} won  F {m(ch["F_change"]):+.5f}', flush=True)


if __name__ == '__main__':
    main()
