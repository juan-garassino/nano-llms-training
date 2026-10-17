# GUARDIAN

Automated security & bug guardian for `juan-garassino/nano-llms-training`.
Work lands only on branch `claude/guardian` via a single draft PR into `main`.

## Checks

| Area | Tool / rule | Scope |
|---|---|---|
| Dependency vulns | `pip-audit -r <requirements>` / `npm audit` | No `requirements*.txt`, `pyproject.toml`, `setup.py` or `package.json` exists -> nothing to audit (see Backlog) |
| Risky code | `bandit -r -q` | `nano_moe/`, `examples/`, `benchmarks/`, `boilerplates/`, `tests/`, `fix_nb.py` |
| Risky code (grep) | `eval(`/`exec(`, `pickle`, `torch.load`, `yaml.load`, `shell=True`, SQL string formatting, `verify=False`, `urlopen/urlretrieve`, CORS, `0.0.0.0` servers | same |
| Secrets | `gitleaks detect --source . --log-opts=--all --redact` (history) + `gitleaks dir . --redact` (tree) | whole repo |
| CI | GitHub Actions pinning / permissions / `${{ github.event.* }}` injection | no `.github/` directory -> N/A |
| Containers | Dockerfile root user / unpinned base | no Dockerfile -> N/A |
| Lint (real errors) | `ruff check --select E9,F63,F7,F82` | all `.py` except `notebooks/` |
| Tests | `pytest -q tests` (CPU torch in a scratch venv) | `tests/` |

**Exclusions:** `notebooks/*.ipynb` (notebook outputs), `input.txt` (Tiny Shakespeare corpus data),
`nano_moe.zip` (archive), `__pycache__/`. `boilerplates/` are standalone, notebook-exported
scripts; they are scanned but reported separately in metrics because two of them do not parse
(`diffusion_clip.py`, `nano_moe_flashattention_mamba_bpe_grpo.py`), which accounts for 55 of the
57 ruff errors.

## Findings ledger

| id | severity | type | file | status | date |
|---|---|---|---|---|---|
| G-001 | high | bug (crash) | `nano_moe/training/trainer.py` `eval_model` unpacked 3 values from models returning 6 (SimpleTransformer, TrueHolographicSFPT, UniversalSFPT) -> `ValueError` on the default `make demo-shakespeare` path | fixed | 2026-10-17 |
| G-002 | medium | bug (crash) | `nano_moe/datasets/text.py` `TextDataset.__len__` over-counted by one window when `len(tokens) % seq_len` is 0 -> short last item, collate error | fixed | 2026-10-17 |
| G-003 | medium | bug (NameError) | `nano_moe/main.py` `universal_phase` used `vocab_size`, undefined for image datasets | fixed | 2026-10-17 |
| G-004 | medium | vuln (B614 unsafe deserialization) | `nano_moe/training/distillation.py:110` `torch.load` of teacher checkpoint without `weights_only=True` | fixed | 2026-10-17 |
| G-005 | low | lint F821 / dead code | `nano_moe/training/trainer.py:49` `logits` referenced via `locals()` before assignment (unused) | fixed | 2026-10-17 |
| G-006 | low | test bug | `tests/test_integration.py` called `test_quantum_diffusion` before definition (F821) and swallowed every exception, so tests could never fail; they also need network | fixed (opt-in via `NANO_MOE_NETWORK_TESTS=1`, exceptions re-raised) | 2026-10-17 |
| G-007a | medium | vuln (B614) | `boilerplates/reflection_layer.py:1248,1388` `torch.load` of user-supplied `--load_path` without `weights_only` | fixed (pass 2) | 2026-10-17 |
| G-007b | medium | vuln (B614, not counted by bandit: file does not parse) | `boilerplates/nano_moe_flashattention_mamba_bpe_grpo.py` (~20 `torch.load` calls) | open | 2026-10-17 |
| G-008 | medium | vuln (B615) | `nano_moe/datasets/text.py:41-57`, `boilerplates/phase_transformer.py:346,349` HF `from_pretrained`/`load_dataset` without revision pin | open | 2026-10-17 |
| G-009 | low | vuln (B310) | `nano_moe/models/quantum_diffusion.py:232`, `boilerplates/reflection_layer.py:578` `urlretrieve` of hard-coded https URLs (no integrity check) | wontfix (constant https URLs) | 2026-10-17 |
| G-010 | low | B603/B607 | `nano_moe/datasets/arc.py:31-32` `subprocess.run(["curl"...])`/`unzip` via PATH, list-args, constant URL | wontfix | 2026-10-17 |
| G-011 | low | B311/B110/B101 | non-crypto `random`, try/except/pass, asserts (29 lows) | wontfix (ML code, not security relevant) | 2026-10-17 |
| G-012 | low | quality | two boilerplate files do not parse (55 ruff E9 errors) | open | 2026-10-17 |
| G-013 | low | hygiene | committed `__pycache__/*.pyc` (22 files, now untracked + ignored), `.DS_Store`, `nano_moe.zip` | partially fixed | 2026-10-17 |
| G-014 | info | deps | no dependency manifest -> pip-audit cannot run, installs unreproducible | open | 2026-10-17 |

Secrets: gitleaks history (12 commits) and working tree -> **no leaks found**.

## Metrics

| metric | before | after (run 1) |
|---|---|---|
| open vulns crit / high / med / low (security) | 0 / 0 / 12 / 29 | 0 / 0 / 9 / 29 (bandit) |
| open bugs (crash/NameError/off-by-one) | 3 | 0 |
| ruff real errors (E9,F63,F7,F82), excl. boilerplates | 2 | 0 |
| ruff real errors incl. boilerplates | 57 | 55 |
| tests collected / passing (offline) | 3 / 3 (all swallowed exceptions; network-bound, cannot fail) | 3 integration (opt-in, skipped by default) + 8 offline regression; not executed in this run: CPU torch wheel could not be installed in time (pytorch CPU index blocked by proxy, PyPI CUDA wheel download did not finish) |
| gitleaks findings | 0 | 0 |

## Backlog

1. Add a `requirements.txt` (torch, torchvision, rich, numpy, matplotlib, tqdm, transformers, datasets) so pip-audit and CI can run.
2. Pin HF revisions (or document) in `nano_moe/datasets/text.py` (G-008).
3. `weights_only=True` on `torch.load` in `boilerplates/nano_moe_flashattention_mamba_bpe_grpo.py` (G-007b).
4. Remove committed `.DS_Store`, `nano_moe.zip` (G-013; `__pycache__` untracked + `.gitignore` extended in run 1).
5. Fix or move to `notebooks/` the two non-parsing boilerplate scripts (G-012).
6. Add a minimal GitHub Actions workflow (pinned by SHA, `permissions: contents: read`) running the offline tests.

## Log

- **Run 1 — 2026-10-17**: initial scan (bandit, gitleaks history+tree, ruff, grep). Fixed G-001..G-006 with offline regression tests in `tests/test_guardian_regressions.py`; made network integration tests opt-in.
- **Run 1, pass 2 — 2026-10-17**: `weights_only=True` for `boilerplates/reflection_layer.py` checkpoint loads (payload is primitives + state_dict); untracked `__pycache__`, extended `.gitignore`.
