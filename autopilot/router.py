"""The router: for every model, the chance it gets this prompt right; then the cheapest
model that is likely enough to.

Features are what a gateway sees before it calls anything: the prompt's embedding, which
product feature sent it (the task), and how long it is. One logistic regression per model
learns P(right | features) from measured outcomes. Routing picks the cheapest model whose
predicted chance clears `confidence`; when none does, the model the router trusts most.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict


@dataclass
class Registry:
    names: list                    # cheapest first
    price: dict                    # name -> (input $ per token, output $ per token)
    frontier: str

    @classmethod
    def from_config(cls, config, only=None):
        models = {n: m for n, m in config["models"].items() if only is None or n in only}
        price = {n: (m["input_per_million"] / 1e6, m["output_per_million"] / 1e6) for n, m in models.items()}
        # cheapest first by the price of a typical request (1,000 tokens in, 200 out)
        names = sorted(models, key=lambda n: price[n][0] * 1000 + price[n][1] * 200)
        return cls(names, price, config["frontier"])

    def cost(self, name, prompt_tokens, output_tokens):
        return prompt_tokens * self.price[name][0] + output_tokens * self.price[name][1]


@dataclass
class Router:
    registry: Registry
    confidence: float = 0.75
    models: dict = field(default_factory=dict)

    def fit(self, x, outcomes):
        """outcomes: {model name: array of 0/1, one per row of x}; rows without a label are NaN."""
        for name in self.registry.names:
            y = outcomes[name]
            seen = ~np.isnan(y)
            if seen.sum() >= 10 and len(set(y[seen])) == 2:
                self.models[name] = LogisticRegression(C=1.0, max_iter=3000).fit(x[seen], y[seen].astype(int))
            else:
                self.models[name] = float(np.nanmean(y)) if seen.any() else 0.5   # too few labels: its base rate
        return self

    def chances(self, x):
        return {n: (m.predict_proba(x)[:, 1] if hasattr(m, "predict_proba") else np.full(len(x), m)) for n, m in self.models.items()}

    def route(self, x, confidence=None):
        """For each row, (model name, why)."""
        confidence = self.confidence if confidence is None else confidence
        p = self.chances(x)
        out = []
        for i in range(len(x)):
            pick = next((n for n in self.registry.names if p[n][i] >= confidence), None)
            if pick:
                out.append((pick, f"cheapest model with a {p[pick][i]:.0%} predicted chance"))
            else:
                best = max(self.registry.names, key=lambda n: p[n][i])
                out.append((best, f"no model reached {confidence:.0%}; best is {p[best][i]:.0%}"))
        return out


def out_of_fold(registry, x, outcomes, folds=5, seed=0):
    """Each training row's predicted chances from a model that never saw it, for tuning."""
    chances = {}
    for name in registry.names:
        clf = LogisticRegression(C=1.0, max_iter=3000)
        # shuffled folds: rows arrive grouped by task, and a fold must not hold out a whole task
        chances[name] = cross_val_predict(clf, x, outcomes[name].astype(int), cv=KFold(folds, shuffle=True, random_state=seed),
                                          method="predict_proba")[:, 1]
    return chances


def pick(registry, chances, confidence):
    """The routing rule on precomputed chances: index of the chosen model per row."""
    names = registry.names
    p = np.vstack([chances[n] for n in names])            # models x rows
    ok = p >= confidence
    first = np.where(ok.any(0), ok.argmax(0), p.argmax(0))
    return np.array(names)[first]
