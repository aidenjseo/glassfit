"""Phase 3 stub — residual-model training CLI.

Plan: per-target sklearn HistGradientBoosting (handles small tabular data, mixed
features, missing values, no tuning needed), trained on the dataset export; a global
model first, then per-user bias offsets once a user has >=5 feedback rows. Models are
saved via ``registry`` and loaded by the backend at startup when compatible with the
current ``engine_version``.
"""

import sys


def train_all() -> None:
    raise NotImplementedError("Phase 3")


if __name__ == "__main__":
    print(__doc__)
    print("Not implemented yet: accumulate feedback rows first (Phase 3).")
    sys.exit(1)
