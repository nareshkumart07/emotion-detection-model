"""Windows-safe LMDB sharing for TorchEEG (same as EDA.ipynb)."""


def apply_lmdb_patch() -> None:
    from torcheeg.io.eeg_signal import LMDBEEGSignalIO

    if getattr(LMDBEEGSignalIO, '_share_env_patched', False):
        return

    def _lmdb_copy_share_env(self):
        result = self.__class__.__new__(self.__class__)
        result.__dict__.update(
            {k: v for k, v in self.__dict__.items() if k != '_env'}
        )
        result._env = self._env
        return result

    LMDBEEGSignalIO.__copy__ = _lmdb_copy_share_env
    LMDBEEGSignalIO.__del__ = lambda self: None
    LMDBEEGSignalIO._share_env_patched = True
