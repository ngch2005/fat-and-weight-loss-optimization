# callbacks.py
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, desc="Training Progress"):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.desc = desc 

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc=self.desc)

    def _on_step(self):
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()