#from pytorch_lightning.callbacks import ModelCheckpoint
#from operator import call

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, MLFlowLogger
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import subprocess
import datetime
import shutil
import socket
import torch
import json
import os

from waveletflux.utils.callbacks import PerformanceTrackingCallback, PortableModelCheckpoint, EmergencySaveCallback

class ExperimentManager:
    """
    A complete experiment manager for multi-scale flow matching experiments.

    Responsibilities:
      - Create and manage experiment directory structure
      - Save configs, environment metadata, and git commit
      - Provide loggers (CSV + TensorBoard + MLFlow)
      - Provide checkpoint callback
      - Save artifacts (figures, tensors, predictions)
      - Autogenerate a README for reproducibility
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, 
                 base_dir: str, 
                 experiment_name: str, 
                 overwrite: bool = False, 
                 loggers: Optional[list] = ['csv'],
                 callbacks: Optional[list] = ['model_checkpoint'],
                 log_every_n_epochs: Optional[int] = 1
                 ):
        """
        Args:
            base_dir (str): Base directory where experiments are stored.
            experiment_name (str): User-defined name of experiment.
            overwrite (bool): If True, delete existing experiment folder.
            loggers (list, optional): List of loggers to use. Options: 'csv', 'mlflow', 'tensorboard'.
        """

        self.base_dir = base_dir
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(base_dir, experiment_name)
        self.loggers = loggers
        self.callbacks = callbacks
        self.log_every_n_epochs = log_every_n_epochs

        # Centralized MLflow directory
        if 'mlflow' in self.loggers:
            self.mlflow_dir = os.path.join(base_dir, "mlflow")
            self.mlflow_db = os.path.join(self.mlflow_dir, "mlflow.db")
            #self.mlflow_artifacts = os.path.join(self.mlflow_dir, "artifacts")

        self.run_name = f"{self.experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if overwrite and os.path.exists(self.experiment_dir):
            shutil.rmtree(self.experiment_dir)

        # Create folder structure
        self._init_directories()
        self._init_readme()

    # ------------------------------------------------------------------
    # Directory Setup
    # ------------------------------------------------------------------

    def _init_directories(self):
        """Create local experiment folder structure and centralized tracking."""
        #os.makedirs(self.experiment_dir, exist_ok=True)

        # Centralized MLflow tracking & artifacts
        #os.makedirs(self.mlflow_dir, exist_ok=True)
        #os.makedirs(self.mlflow_artifacts, exist_ok=True)
        
        os.makedirs(self.experiment_dir, exist_ok=True)
        if 'mlflow' in self.loggers:
            os.makedirs(self.mlflow_dir, exist_ok=True)

        # Local experiment directories
        self.dirs = {
            "config"        :   os.path.join(self.experiment_dir, "config"),
            "env"           :   os.path.join(self.experiment_dir, "environment"),
            "logs"          :   os.path.join(self.experiment_dir, "logs"),
            "artifacts"     :   os.path.join(self.experiment_dir, "artifacts"),
            "checkpoints"   :   os.path.join(self.experiment_dir, "artifacts/checkpoints"),
            "figures"       :   os.path.join(self.experiment_dir, "artifacts/figures"),
            "samples"       :   os.path.join(self.experiment_dir, "artifacts/samples"),
            "performance"   :   os.path.join(self.experiment_dir, "performance"),
        }

        if 'tensorboard' in self.loggers:
            self.dirs["tensorboard"] = os.path.join(self.experiment_dir, "logs/tensorboard")
        if 'csv' in self.loggers:
            self.dirs["csv"] = os.path.join(self.experiment_dir, "logs/csv")

        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)

    # ------------------------------------------------------------------
    # Config Saving
    # ------------------------------------------------------------------

    def save_config(self, config_dict: Dict[str, Any], name: str = "config.json"):
        """
        Save dict-based config as JSON into the config/ folder.

        Args:
            config_dict: Dictionary of config parameters.
            name: Filename inside config folder.
        """
        path = os.path.join(self.dirs["config"], name)
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=4)
        return path

    def copy_config_file(self, src: str):
        """
        Copy a TOML/YAML/JSON config file into config/ folder.
        """
        shutil.copy(src, self.dirs["config"])

    # ------------------------------------------------------------------
    # Environment Logging
    # ------------------------------------------------------------------

    def save_environment_info(self):
        """Save pip freeze, git commit, and hardware details."""

        # Save pip freeze
        try:
            freeze = subprocess.check_output(["pip", "freeze"]).decode()
        except Exception:
            freeze = "pip freeze failed"
        with open(os.path.join(self.dirs["env"], "pip_freeze.txt"), "w") as f:
            f.write(freeze)

        # Save git commit
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except Exception:
            commit = "no git repo or git command failed"
        with open(os.path.join(self.dirs["env"], "git_commit.txt"), "w") as f:
            f.write(commit)

        # Save hardware info
        hw_path = os.path.join(self.dirs["env"], "hardware.txt")
        with open(hw_path, "w") as f:
            f.write(f"Hostname: {socket.gethostname()}\n")
            f.write(f"Date: {str(datetime.datetime.now())}\n\n")
            if torch.cuda.is_available():
                prop = torch.cuda.get_device_properties(0)
                f.write(f"GPU: {prop.name}\n")
                f.write(f"Memory: {prop.total_memory/1024**3:.2f} GB\n")
            else:
                f.write("Running on CPU\n")

    # ------------------------------------------------------------------
    # Loggers: CSV + TensorBoard + MLFlow
    # ------------------------------------------------------------------

    def get_loggers(self):
        """
        Returns CSVLogger and/or TensorBoardLogger and/or MLFlowLogger configured for the experiment.
        """
        loggers = []

        # MLFlow Logger
        if 'mlflow' in self.loggers:
            tracking_uri = f"sqlite:///{self.mlflow_db}"

            mlflow_logger = MLFlowLogger(
                experiment_name = self.experiment_name,
                run_name = self.run_name,
                tracking_uri = tracking_uri,
                log_model = False,
            )
            loggers.append(mlflow_logger)

        # TensorBoard Logger
        if 'tensorboard' in self.loggers:
            tb_logger = TensorBoardLogger(
                save_dir = self.dirs["tensorboard"],
                name = self.run_name
            )
            loggers.append(tb_logger)

        if 'csv' in self.loggers:
            # CSV Logger
            csv_logger = CSVLogger(
                save_dir = self.dirs["csv"],
                name = self.run_name
            )
            loggers.append(csv_logger)

        return loggers

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def get_performance_callback(self, perf_dir: str):
        return PerformanceTrackingCallback(perf_dir)

    def get_emergency_callback(self):
        return EmergencySaveCallback()
    
    def get_model_checkpoint_callback(self, ckpt_dir: str, monitor: str = "val_loss"):
        #return ModelCheckpoint(
        return PortableModelCheckpoint(
            dirpath = ckpt_dir,
            monitor = monitor,
            mode = "min",
            save_top_k = 1,
            save_last = False,
            filename = "best",
            auto_insert_metric_name = False,
            enable_version_counter = False,
            )
    
    #def get_portable_checkpoint_callback(self, ckpt_dir: str, every_n_epochs: int = 5):
    #    return PortableCheckpointCallback(
    #        dirpath = ckpt_dir,
    #        every_n_epochs = every_n_epochs,
    #    )

    def get_callbacks(self, checkpoint_monitor: str = "val_loss"):
        """
        Returns ModelCheckpoint callback configured for the experiment.
        """
        callbacks = []
       
        if 'model_checkpoint_cb' in self.callbacks:
            callbacks.append(
                self.get_model_checkpoint_callback(
                    self.dirs["checkpoints"],
                    monitor=checkpoint_monitor,
                )
            )
            #model_ckpt = self.get_model_checkpoint_callback(self.dirs["checkpoints"])
            #callbacks.append(model_ckpt)

        if 'emergency_cb' in self.callbacks:
            callbacks.append(self.get_emergency_callback())
            #emergency_ckpt = self.get_emergency_callback()
            #callbacks.append(emergency_ckpt)

        if 'performance_cb' in self.callbacks:
            callbacks.append(self.get_performance_callback(self.dirs["performance"]))
            #performance_ckpt = self.get_performance_callback(self.dirs["performance"])
            #callbacks.append(performance_ckpt)

        #if 'portable_checkpoint_cb' in self.callbacks:
        #    portable_ckpt = self.get_portable_checkpoint_callback(self.dirs["checkpoints"], self.log_every_n_epochs)
        #    callbacks.append(portable_ckpt)

        return callbacks

    # ------------------------------------------------------------------
    # Saving Artifacts
    # ------------------------------------------------------------------

    def save_figure(self, fig, name: str):
        """
        Save a matplotlib figure in the artifacts/figures directory.

        Args:
            fig: Matplotlib figure object.
            name: Filename (e.g., "epoch_1_prediction.png")
        """
        fig_path = os.path.join(self.dirs["figures"], name)
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        return fig_path

    def save_tensor(self, tensor: torch.Tensor, name: str):
        """
        Save a tensor (e.g., predictions, wavelet coeffs).
        """
        path = os.path.join(self.dirs["samples"], name)
        torch.save(tensor.cpu(), path)
        return path
    
    def dump_sample_plots(self, yhat: torch.Tensor, y: torch.Tensor, field_names: list, fig_name: str, cmap: str = "RdBu_r", dpi: int = 300):
        """
        Save simple comparison plots between predicted and true vector fields.

        Args:
            yhat: Predicted vector field (B, C, T_out, H, W)
            y: True vector field (B, C, T_out, H, W)
            fig_name: Figure name
            cmap: Colormap for plotting
        """
        fig = plt.figure(figsize=(15, 6))

        shrink = 0.35

        B, C, _, _, _ = yhat.shape

        # Random batch index to plot a random sample in the batch
        batch_idx = torch.randint(0, B, (1,)).item()

        vmin_vals = []
        vmax_vals = []
        
        # Reference
        for c in range(C):
            vmin_vals.append(y[batch_idx, c, 0, :, :].cpu().min().item())
            vmax_vals.append(y[batch_idx, c, 0, :, :].cpu().max().item())
            
            plt.subplot(3, C, c + 1)
            plt.imshow(y[batch_idx, c, 0, :, :].cpu(), vmin=vmin_vals[c], vmax=vmax_vals[c], cmap=cmap)
            plt.colorbar(shrink=shrink)
            plt.title(f"{field_names[c]} (ref)")
            plt.axis('off')

        # Predicted
        for c in range(C):
            plt.subplot(3, C, C + c + 1)
            plt.imshow(yhat[batch_idx, c, 0, :, :].cpu(), vmin=vmin_vals[c], vmax=vmax_vals[c], cmap=cmap)
            plt.colorbar(shrink=shrink)
            plt.title(f"{field_names[c]} (pred)")
            plt.axis('off')

        # Difference (Bias/Error)
        for c in range(C):
            plt.subplot(3, C, 2 * C + c + 1)
            plt.imshow(y[batch_idx, c, 0, :, :].cpu() - yhat[batch_idx, c, 0, :, :].cpu(), cmap=cmap)
            plt.colorbar(shrink=shrink)
            plt.axis('off')

        dst_path = os.path.join(self.dirs["figures"], fig_name)
        plt.savefig(dst_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)

    # ------------------------------------------------------------------
    # README Generation
    # ------------------------------------------------------------------

    def _init_readme(self):
        path = os.path.join(self.experiment_dir, "README.md")
        with open(path, "w") as f:
            f.write(f"# Experiment: {self.experiment_name}\n")
            f.write(f"Run name: `{self.run_name}`\n")
            f.write(f"Created: {str(datetime.datetime.now())}\n\n")
            f.write("## Folder Structure\n")
            for key, p in self.dirs.items():
                f.write(f"- `{key}` → `{p}`\n")

    def append_readme(self, text: str):
        with open(os.path.join(self.experiment_dir, "README.md"), "a") as f:
            f.write(text + "\n")

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def path(self, key: str):
        """Get full path to a subdirectory by key."""
        return self.dirs.get(key, None)
