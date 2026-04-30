import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

sys.stdout.flush()

# -----------------------------
# Config
# -----------------------------
SEED = 42
N = 60000
TARGET_SHARE = 0.50          # neutral target; change if you want a different overall balance
BASE_CATEGORY_WEIGHT = 0.45  # controls how strong each category is in the synthetic vote model
NOISE_SD = 0.22              # small noise for realism, but not so large that it drowns out signal

# Define which category pairs should be crossed for interactions
INTERACTION_PAIRS = [
    ("race", "age"),
    ("race", "gender"),
    ("race", "state")
]

# Inject true latent interactions into the synthetic data so the model has something to find!
# Positive = more Dem, Negative = more Rep
TRUE_INTERACTIONS = {
    "race__age": {
        "Black": {
            "18–29": -0.20,
            "30–44": -0.07,
            "45–64": 0.47,
            "65+": 0.62
        }
    },
    "race__gender": {
        "Black": {
            "Female": 0.35,  # Black women trend more heavily Dem than the additive main effects suggest
            "Male": -0.25
        },
        "White": {
            "Female": 0.15,
            "Male": -0.15
        },
        "Latino": {
            "Male": -0.20    # Reflecting recent rightward shifts among Latino men
        }
    },
    "race__state": {
        "White": {
            "California": 0.25,
            "Texas": -0.25,
            "New York": 0.20,
            "Alabama": -0.30
        },
        "Latino": {
            "Florida": -0.35, # Unique Latino dynamics in FL
            "Texas": -0.15,
            "California": 0.15
        }
    }
}

rng = np.random.default_rng(SEED)

# -----------------------------
# Helpers
# -----------------------------
def sigmoid(x):
    x = np.clip(x, -35, 35)
    return 1.0 / (1.0 + np.exp(-x))

def logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))

def normalize_dict(d):
    vals = np.array(list(d.values()), dtype=float)
    vals = np.clip(vals, 1e-12, None)
    vals = vals / vals.sum()
    return {k: float(v) for k, v in zip(d.keys(), vals)}

def get_root_bracket(base_scores, target):
    lo, hi = -6.0, 6.0

    def mean_prob(b):
        return float(sigmoid(base_scores + b).mean())

    # Expand until the target is bracketed
    while mean_prob(lo) > target:
        lo -= 2.0
    while mean_prob(hi) < target:
        hi += 2.0

    return lo, hi, mean_prob

# -----------------------------
# Load data
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "voter-dna-data.json")

print("Loading data...", flush=True)
with open(file_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

demographics = raw["demographics"]
category_order = raw["categoryOrder"]

# -----------------------------
# Population priors
# -----------------------------
POP = {
    "race": {
        "White": 0.633, "Black": 0.115, "Latino": 0.135,
        "Asian": 0.048, "Others": 0.069
    },
    "gender": {
        "Male": 0.478, "Female": 0.522
    },
    "sexuality": {
        "Straight": 0.945, "Gay/Lesbian": 0.055
    },
    "area": {
        "Urban": 0.310, "Suburban": 0.490, "Rural": 0.200
    },
    "religion": {
        "Evangelical": 0.246, "Protestant": 0.148, "Catholic": 0.196,
        "Jewish": 0.022, "Mormons": 0.018, "Muslim": 0.018,
        "Other religion": 0.078, "No religion": 0.274
    }
}

# Fill or normalize every category's population distribution
for cat in category_order:
    opts = list(demographics[cat]["options"].keys())
    if cat not in POP:
        POP[cat] = {k: 1.0 / len(opts) for k in opts}
    else:
        sub = {k: POP[cat].get(k, 1.0 / len(opts)) for k in opts}
        total = sum(sub.values())
        POP[cat] = {k: v / total for k, v in sub.items()}

# State distribution: use a few large states, then spread the rest evenly
_states = list(demographics["state"]["options"].keys())
_big = {
    "California": 0.117, "Texas": 0.082, "Florida": 0.059, "New York": 0.057,
    "Pennsylvania": 0.040, "Illinois": 0.038, "Ohio": 0.034, "Georgia": 0.031,
    "North Carolina": 0.030, "Michigan": 0.030, "New Jersey": 0.027,
    "Virginia": 0.024, "Washington": 0.022, "Arizona": 0.022, "Massachusetts": 0.020
}
remaining_mass = 1.0 - sum(_big.values())
remaining_states = [s for s in _states if s not in _big]
rest_share = remaining_mass / len(remaining_states) if remaining_states else 0.0
POP["state"] = {s: _big.get(s, rest_share) for s in _states}
POP["state"] = normalize_dict(POP["state"])

# -----------------------------
# Build mathematically centered effect tables
# -----------------------------
effect_table = {}
category_scale = {}

for cat in category_order:
    opts = list(demographics[cat]["options"].keys())

    pop_w = np.array([POP[cat].get(opt, 0.0) for opt in opts], dtype=float)
    pop_w = pop_w / pop_w.sum()

    raw_scores = np.array(
        [logit(demographics[cat]["options"][opt]["dem"] / 100.0) for opt in opts],
        dtype=float
    )

    weighted_mean = float(np.sum(pop_w * raw_scores))
    centered = raw_scores - weighted_mean
    weighted_sd = float(np.sqrt(np.sum(pop_w * centered**2)))

    scale = BASE_CATEGORY_WEIGHT / (1.0 + weighted_sd)

    effect_table[cat] = {opt: float(centered[i]) for i, opt in enumerate(opts)}
    category_scale[cat] = float(scale)

# -----------------------------
# Sample synthetic voters
# -----------------------------
def sample_option(cat):
    opts = list(POP[cat].keys())
    probs = np.array(list(POP[cat].values()), dtype=float)
    probs = probs / probs.sum()
    return rng.choice(opts, p=probs)

print(f"Generating {N} voters...", flush=True)

rows = []
for _ in range(N):
    v = {cat: sample_option(cat) for cat in category_order}
    rows.append(v)

df = pd.DataFrame(rows)

# Base latent score without intercept (Main Effects)
base_scores = np.zeros(len(df), dtype=float)
for cat in category_order:
    base_scores += df[cat].map(lambda x: category_scale[cat] * effect_table[cat][x]).to_numpy(dtype=float)

# Inject True Interactions into the latent score
for pair_key, rules in TRUE_INTERACTIONS.items():
    c1, c2 = pair_key.split("__")
    for v1, subrules in rules.items():
        for v2, effect in subrules.items():
            mask = (df[c1] == v1) & (df[c2] == v2)
            base_scores[mask] += effect

# Add a small amount of idiosyncratic noise
base_scores += rng.normal(0.0, NOISE_SD, size=len(df))

# Calibrate intercept so overall mean vote rate hits the target
lo, hi, mean_prob = get_root_bracket(base_scores, TARGET_SHARE)

for _ in range(70):
    mid = (lo + hi) / 2.0
    m = mean_prob(mid)
    if m < TARGET_SHARE:
        lo = mid
    else:
        hi = mid

intercept = (lo + hi) / 2.0
probs = sigmoid(base_scores + intercept)

df["vote"] = rng.binomial(1, probs)

print(f"Target share={TARGET_SHARE:.3f}  Actual share={df['vote'].mean():.3f}", flush=True)

# -----------------------------
# Fit LASSO on the synthetic voters
# -----------------------------
X_raw = pd.get_dummies(df[category_order], drop_first=False).astype(float)

# Append interaction features to the dataframe
for c1, c2 in INTERACTION_PAIRS:
    for opt1 in demographics[c1]["options"]:
        for opt2 in demographics[c2]["options"]:
            col1 = f"{c1}_{opt1}"
            col2 = f"{c2}_{opt2}"
            if col1 in X_raw.columns and col2 in X_raw.columns:
                X_raw[f"{c1}__{c2}__{opt1}__{opt2}"] = X_raw[col1] * X_raw[col2]

y = df["vote"].to_numpy(dtype=int)
fnames = X_raw.columns.tolist()

print(f"Features={len(fnames)} (Main + Interactions)", flush=True)

scaler = StandardScaler(with_mean=False)
Xs = scaler.fit_transform(X_raw)

print("Fitting LASSO...", flush=True)
clf = LogisticRegressionCV(
    Cs=np.logspace(-2, 1.5, 20),
    cv=5,
    penalty="l1",
    solver="saga",
    fit_intercept=False,
    max_iter=4000,
    random_state=SEED,
    n_jobs=-1,
    scoring="accuracy"
)
clf.fit(Xs, y)

coefs = clf.coef_[0]
pred_acc = float((clf.predict(Xs) == y).mean())

print(
    f"C={clf.C_[0]:.4f} "
    f"nonzero={int(np.sum(np.abs(coefs) > 1e-8))} "
    f"acc={pred_acc:.3f}",
    flush=True
)

# -----------------------------
# Convert fitted coefficients back to option-level raw effects
# -----------------------------
sigma = scaler.scale_
raw_d = {}

# 1. Main Effects
for cat in category_order:
    raw_d[cat] = {}
    for opt in demographics[cat]["options"]:
        fn = f"{cat}_{opt}"
        if fn in fnames:
            idx = fnames.index(fn)
            beta = float(coefs[idx])
            sig = float(sigma[idx])
            raw_d[cat][opt] = beta / sig if abs(sig) > 1e-12 else 0.0
        else:
            raw_d[cat][opt] = 0.0

cat_mean = {}
for cat in category_order:
    cat_mean[cat] = sum(
        POP[cat].get(opt, 0.0) * raw_d[cat][opt]
        for opt in demographics[cat]["options"]
    )

option_deltas = {}
zeroed = []

for cat in category_order:
    option_deltas[cat] = {}
    for opt in demographics[cat]["options"]:
        d = raw_d[cat][opt] - cat_mean[cat]
        d = float(round(d, 6))
        option_deltas[cat][opt] = d
        if abs(d) < 1e-6:
            zeroed.append(f"{cat}_{opt}")

all_d = [
    (f"{cat}={opt}", option_deltas[cat][opt])
    for cat in category_order
    for opt in demographics[cat]["options"]
]
all_d.sort(key=lambda x: abs(x[1]), reverse=True)

# 2. Interaction Effects Extraction
interaction_deltas = {
    "_comment": "Adjustments applied ON TOP of main effects when two conditions are both selected. Positive = more Dem, negative = more Rep."
}

for c1, c2 in INTERACTION_PAIRS:
    pair_key = f"{c1}__{c2}"
    interaction_deltas[pair_key] = {}
    for opt1 in demographics[c1]["options"]:
        interaction_deltas[pair_key][opt1] = {}
        for opt2 in demographics[c2]["options"]:
            fn = f"{c1}__{c2}__{opt1}__{opt2}"
            if fn in fnames:
                idx = fnames.index(fn)
                beta = float(coefs[idx])
                sig = float(sigma[idx])
                # Unscale the coefficient directly; we don't center interactions to keep them as pure "add-ons"
                val = beta / sig if abs(sig) > 1e-12 else 0.0
                interaction_deltas[pair_key][opt1][opt2] = float(round(val, 6))
            else:
                interaction_deltas[pair_key][opt1][opt2] = 0.0

print("Top 25 marginal deltas:")
for label, d in all_d[:25]:
    side = "D" if d > 0 else "R"
    print(f"  {side} {d:+.4f}  {label}")
print(f"Zeroed: {len(zeroed)}", flush=True)

# -----------------------------
# Save output
# -----------------------------
out = {
    "_meta": {
        "model": "Logistic L1 LASSO marginal + interactions",
        "n_synthetic": int(N),
        "seed": int(SEED),
        "target_share": round(float(TARGET_SHARE), 6),
        "actual_share": round(float(df["vote"].mean()), 6),
        "intercept": round(float(intercept), 6),
        "noise_sd": round(float(NOISE_SD), 6),
        "base_category_weight": round(float(BASE_CATEGORY_WEIGHT), 6),
        "best_C": round(float(clf.C_[0]), 6),
        "lambda": round(1.0 / float(clf.C_[0]), 6),
        "n_nonzero": int(np.sum(np.abs(coefs) > 1e-8)),
        "n_total_feats": int(len(fnames)),
        "train_accuracy": round(pred_acc, 4)
    },
    "option_deltas": option_deltas,
    "zeroed_features": zeroed,
    "interaction_deltas": interaction_deltas
}

out_path = os.path.join(base_dir, "lasso-weights.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print(f"DONE -> {out_path}", flush=True)