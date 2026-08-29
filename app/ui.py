"""
RunLog: a progress bar with a running log of what's actually happening.

A bare percentage on a multi-stage pipeline is uninformative - "47%" doesn't say whether the
model is clustering, normalizing batch 3 of 5, or writing copy for product 12. Each stage here
costs a round-trip, so when something is slow or stalls, the log is what tells you where.

Wraps st.status: the label updates live, lines accumulate inside, and the whole thing collapses
to a single tick when the run finishes. Failures stay expanded with the error in place.
"""

import time

import streamlit as st


class RunLog:
    def __init__(self, label: str, expanded: bool = True):
        self.status = st.status(label, expanded=expanded)
        self.bar = self.status.progress(0.0)
        self.started = time.time()
        self.count = 0

    def _elapsed(self) -> str:
        seconds = int(time.time() - self.started)
        return f"{seconds // 60}:{seconds % 60:02d}"

    def step(self, fraction: float, text: str):
        """Advance the bar and append a line to the log."""
        self.count += 1
        fraction = min(max(fraction, 0.0), 1.0)
        self.bar.progress(fraction, text=f"{int(fraction * 100)}% · {text}")
        self.status.write(f"`{self._elapsed()}`  {text}")

    def note(self, text: str):
        """Log a line without moving the bar - results, counts, warnings."""
        self.status.write(f"`{self._elapsed()}`  {text}")

    def done(self, label: str | None = None):
        self.bar.progress(1.0, text="100% · done")
        self.status.update(
            label=label or f"Done in {self._elapsed()}",
            state="complete", expanded=False,
        )

    def fail(self, text: str):
        self.status.write(f"`{self._elapsed()}`  ❌ {text}")
        self.status.update(label=f"Failed after {self._elapsed()}", state="error", expanded=True)
