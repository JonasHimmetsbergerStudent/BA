"""
Brain FID – minimales Beispiel
==============================
Zeigt Schritt für Schritt wie FID berechnet wird,
passend zu deinem SpikeGPT-Code.
"""

import numpy as np
from scipy.linalg import sqrtm


# -----------------------------------------------------------------------
# Schritt 1: Features extrahieren
# -----------------------------------------------------------------------
# In deinem echten Code würdest du hier das Modell benutzen:
#
#   h = model(x.bfloat16(), block_mask, positions).float()  # (B, T, d)
#   features = h.mean(dim=1)  # (B, d)  — einen Vektor pro Sequenz
#
# Hier simulieren wir das mit zufälligen Zahlen:

d = 384  # n_embd aus deinem CONFIGS["tiny"]

# Echte Spikes → durch Modell → Features
real_features = np.random.randn(500, d)  # 500 echte Sequenzen

# Generierte Spikes → durch Modell → Features
# (gut trainiertes Modell: ähnlich verteilt wie real_features)
generated_features = np.random.randn(500, d) + 0.1  # leicht verschoben


# -----------------------------------------------------------------------
# Schritt 2: Mittelwert und Kovarianz berechnen
# -----------------------------------------------------------------------

mu_r = real_features.mean(axis=0)        # (d,)
mu_g = generated_features.mean(axis=0)   # (d,)

sigma_r = np.cov(real_features, rowvar=False)   # (d, d)
sigma_g = np.cov(generated_features, rowvar=False)  # (d, d)

# -----------------------------------------------------------------------
# Schritt 3: FID berechnen
# -----------------------------------------------------------------------

def compute_fid(mu_r, mu_g, sigma_r, sigma_g):
    """
    FID = ||mu_r - mu_g||^2 + tr(sigma_r + sigma_g - 2 * sqrt(sigma_r @ sigma_g))

    mu_r, mu_g    : Mittelwert-Vektoren   (d,)
    sigma_r, sigma_g : Kovarianzmatrizen  (d, d)
    """

    # --- Teil 1: Abstand der Mittelwerte ---
    diff = mu_r - mu_g
    mean_dist = diff @ diff  # ||mu_r - mu_g||^2 (ein einziger Skalar)

    # --- Teil 2: Abstand der Kovarianzmatrizen ---
    # Matrixwurzel von (sigma_r @ sigma_g)
    product = sigma_r @ sigma_g
    sqrt_product = sqrtm(product)  # das erledigt scipy für uns

    # Imaginärteil entsteht durch Rundungsfehler → einfach wegwerfen
    if np.iscomplexobj(sqrt_product):
        sqrt_product = sqrt_product.real

    cov_dist = np.trace(sigma_r + sigma_g - 2 * sqrt_product)

    fid = mean_dist + cov_dist
    return fid, mean_dist, cov_dist


fid, mean_dist, cov_dist = compute_fid(mu_r, mu_g, sigma_r, sigma_g)

print(f"FID gesamt  : {fid:.2f}")
print(f"  Teil 1 (Mittelwert-Abstand)  : {mean_dist:.2f}")
print(f"  Teil 2 (Kovarianz-Abstand)   : {cov_dist:.2f}")


# -----------------------------------------------------------------------
# Schritt 4: Was bedeutet der Wert?
# -----------------------------------------------------------------------

print()
print("Zum Vergleich:")

# Perfekter Fall: gleiche Verteilung
perfect_features = np.random.randn(500, d)
fid_perfect, _, _ = compute_fid(
    perfect_features.mean(axis=0),
    perfect_features.mean(axis=0),
    np.cov(perfect_features, rowvar=False),
    np.cov(perfect_features, rowvar=False),
)
print(f"  FID (identisch)   : {fid_perfect:.2f}  ← sollte ~0 sein")

# Schlechter Fall: komplett andere Verteilung
bad_features = np.random.randn(500, d) * 3 + 5  # weit verschoben
fid_bad, _, _ = compute_fid(
    real_features.mean(axis=0),
    bad_features.mean(axis=0),
    np.cov(real_features, rowvar=False),
    np.cov(bad_features, rowvar=False),
)
print(f"  FID (komplett anders): {fid_bad:.2f}  ← viel größer")


# -----------------------------------------------------------------------
# Wie du das in deinen Trainingscode einbauen würdest
# -----------------------------------------------------------------------
#
# In _run_validation() hast du schon:
#
#   sample = _sample_autoregressive(model, adapter, rasters, ...)
#
# Du bräuchtest dann:
#
# 1) real_feats   = encode_sequences(model, adapter, real_windows)
# 2) gen_feats    = encode_sequences(model, adapter, generated_windows)
# 3) fid = compute_fid(real_feats.mean(0), gen_feats.mean(0),
#                      np.cov(real_feats, rowvar=False),
#                      np.cov(gen_feats, rowvar=False))
# 4) all_metrics["brain_fid"] = fid
#
# Das encode_sequences wäre einfach:
#
#   def encode_sequences(model, adapter, windows, session, device):
#       feats = []
#       for batch in windows:
#           x = encode(batch, adapter, session)
#           h = model(x, block_mask, positions)  # (B, T, d)
#           feats.append(h.mean(dim=1).cpu().numpy())  # (B, d)
#       return np.concatenate(feats, axis=0)  # (N, d)