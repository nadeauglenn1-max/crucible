# Contributing to Crucible

Crucible is open source under the [MIT License](LICENSE) — fork it, build on it, ship
your own environments. Contributions back to this repository are welcome, with one
governing rule.

## Every merge needs a maintainer +1

- `master` is protected — no direct pushes.
- Propose changes as a pull request from a fork or branch.
- [`.github/CODEOWNERS`](.github/CODEOWNERS) routes every path to the maintainer
  (@nadeauglenn1-max), so with branch protection set to *require review from Code
  Owners*, approval is enforced by the repo, not just by convention.
- The maintainer has final say on what merges.

## The bar (same as every change here)

1. **Tests ship with the change**, and coverage stays **≥ 90%** (`pytest` enforces
   it; CI runs the gate on Python 3.11–3.13).
2. **Docs move with the code** — update the README / `docs/` in the *same* PR that
   changes behavior. New feature ⇒ its "how" in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
   or a backlog entry in [`docs/BACKLOG.md`](docs/BACKLOG.md).
3. **The core stays zero-dependency.** Environments and agents may add dependencies;
   `crucible/` core (env, trajectory, rollout, cli) imports only the standard library.
4. **Honor the determinism contract.** An environment must be a pure function of its
   seed and actions, or `replay` will (correctly) fail it.
5. **Stay inside the boundary.** Crucible is training/eval infrastructure, not runtime
   agent-accountability — see [`docs/VISION.md`](docs/VISION.md) §5.

## Getting started

```bash
pip install -e ".[dev]"
pytest
```

New environments are the most valuable contribution — see the recipe in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §7. By contributing, you agree your
contribution is provided under the project's MIT License.
