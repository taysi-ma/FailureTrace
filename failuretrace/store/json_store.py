"""Write-once raw trial JSON under ``<data_dir>/trials/`` (one file per trial).

Files are immutable: an attempt to overwrite an existing trial raises
:class:`DuplicateRecordError`. Writes are atomic (temp file + rename).
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..core.models import TrialRecord
from ..core.settings import Settings
from .errors import DuplicateRecordError

logger = logging.getLogger(__name__)


class JsonStore:
    def __init__(self, settings: Settings) -> None:
        self.trials_dir = Path(settings.paths.data_dir) / "trials"

    def _path(self, trial_id: str) -> Path:
        return self.trials_dir / f"{trial_id}.json"

    def exists(self, trial_id: str) -> bool:
        return self._path(trial_id).exists()

    def write_trial(self, rec: TrialRecord) -> Path:
        self.trials_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(rec.trial_id)
        if path.exists():
            raise DuplicateRecordError(f"trial JSON already exists (write-once): {path}")
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(rec.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)
        logger.debug("wrote raw trial JSON %s", path)
        return path

    def read_trial(self, trial_id: str) -> TrialRecord:
        return TrialRecord.model_validate_json(
            self._path(trial_id).read_text(encoding="utf-8")
        )
