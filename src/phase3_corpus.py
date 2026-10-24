"""Phase 3 — Deux corpus de niveau caractère, déterministes (graines fixées).

Tâche A : texte « français » synthétique — mots à syllabes consonne/voyelle (vocabulaire de 300
mots tiré d'une loi de Zipf), phrases de 5-11 mots, ponctuation , et .
Tâche B : code arithmétique synthétique — lignes « fn_d(x) = x * d1 + d2 », « r_d = fn_d(x) - x »,
avec chiffres et opérateurs ; structure plus rigide, alphabet plus petit.

Les deux tasks sont des modèles de langage au niveau caractère (prédire le caractère suivant).
"""
import numpy as np

CONS = list("bcdfglmnprstv")
VOW = list("aeiou")


def make_corpus_a(n_chars, seed=0):
    rng = np.random.default_rng(seed)
    words = []
    for _ in range(300):
        nsyl = rng.integers(1, 4)
        w = "".join(rng.choice(CONS) + rng.choice(VOW) for _ in range(nsyl))
        if rng.random() < 0.3:
            w += rng.choice(CONS)
        words.append(w)
    weights = 1.0 / (np.arange(len(words)) + 5.0)
    weights /= weights.sum()
    out = []
    n = 0
    while n < n_chars:
        ns = rng.integers(5, 12)
        sent = []
        for i in range(ns):
            sent.append(words[rng.choice(len(words), p=weights)])
            if i < ns - 1 and rng.random() < 0.08:
                sent[-1] += ","
        out.append(" ".join(sent) + ".")
        n += ns * 6
    text = " ".join(out)
    return text[:n_chars]


def make_corpus_b(n_chars, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    n = 0
    while n < n_chars:
        d1, d2 = rng.integers(0, 50, 2)
        kind = rng.integers(0, 3)
        if kind == 0:
            line = f"fn_{d1}(x) = x * {d2} + {d1}\n"
        elif kind == 1:
            line = f"r_{d1} = fn_{d2}(x) - x\n"
        else:
            line = f"x = {d1} * {d2} + x\n"
        out.append(line)
        n += len(line)
    return "".join(out)[:n_chars]


class CharData:
    """Encode un texte en codes + fenêtres d'échantillonnage / d'évaluation déterministes."""

    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab = chars
        self.code = {c: i for i, c in enumerate(chars)}
        self.V = len(chars)
        self.data = np.array([self.code[c] for c in text], dtype=np.int64)

    def sample_batch(self, B, T, rng):
        ix = rng.integers(0, len(self.data) - T - 1, size=B)
        return np.stack([self.data[i:i + T] for i in ix])

    def eval_windows(self, T, n_windows, seed=123):
        rng = np.random.default_rng(seed)
        ix = np.linspace(0, len(self.data) - T - 1, n_windows).astype(int)
        return np.stack([self.data[i:i + T] for i in ix])

    @classmethod
    def from_codes(cls, codes, vocab):
        """Construit un CharData à partir de codes et d'un vocabulaire partagés (split train/val)."""
        obj = cls.__new__(cls)
        obj.vocab, obj.code, obj.V, obj.data = vocab, {c: i for i, c in enumerate(vocab)}, len(vocab), codes
        return obj

    def perturbed(self, frac=0.1, seed=7):
        """Copie avec une fraction des caractères remplacés aléatoirement (robustesse au bruit)."""
        rng = np.random.default_rng(seed)
        d = self.data.copy()
        n = int(frac * len(d))
        idx = rng.choice(len(d), size=n, replace=False)
        d[idx] = rng.integers(0, self.V, size=n)
        return CharData.from_codes(d, self.vocab)


def get_task(task, train_chars=600_000, val_frac=0.1, seed=0):
    """Un seul corpus par tâche (mêmes mots, même distribution), split 90 % / 10 %."""
    total = int(train_chars * (1 + val_frac))
    if task == "a":
        text = make_corpus_a(total, seed=seed)
    elif task == "b":
        text = make_corpus_b(total, seed=seed)
    else:
        raise ValueError(task)
    cd = CharData(text)
    n_train = int(len(cd.data) * (1 - val_frac))
    train = CharData.from_codes(cd.data[:n_train], cd.vocab)
    val = CharData.from_codes(cd.data[n_train:], cd.vocab)
    return train, val
