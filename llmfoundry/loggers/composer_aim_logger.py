######## WARNING: THIS IS AN INITIAL DRAFT GENERATED WITH AI ASSISTANCE. THIS NEEDS TO BE REVIEWED AND TESTED. ########

import atexit
import os
from logging import getLogger
from typing import Dict, Optional, Any, Sequence, Union

import numpy as np
import torch

from aim.ext.resource.configs import DEFAULT_SYSTEM_TRACKING_INT
from aim.sdk.run import Run

from llmfoundry.loggers.aim_remote_uploader import upload_repo

try:
    from composer.core import State
    from composer.loggers import Logger, LoggerDestination
except ImportError:
    raise RuntimeError(
        'This contrib module requires composer to be installed. ' +
        'Please install it with command: \n pip install mosaicml'
    )

sys_logger = getLogger(__name__)


def _get_global_rank() -> int:
    """Return the global rank for distributed training.
    If torch.distributed is not available or not initialized, returns 0."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class AimLogger(LoggerDestination):
    """Logger for tracking MosaicML Composer training with Aim.

    Args:
        repo (str, optional): Path or URI for the Aim repository.
        experiment_name (str, optional): The Aim experiment name.
        system_tracking_interval (int, optional): Frequency for system-level tracking.
        log_system_params (bool): Whether to log system-level metrics (e.g. CPU usage).
        capture_terminal_logs (bool, optional): Whether to capture terminal (stdout) logs.
        rank_zero_only (bool): If True, only the global rank 0 logs metrics.
        entity (str, optional): For organizational purposes (not strictly used by Aim).
        project (str, optional): For organizational purposes (not strictly used by Aim).
        upload_on_close (bool): Whether to upload the Aim repo on trainer close.
    """

    def __init__(
        self,
        repo: Optional[str] = None,
        experiment_name: Optional[str] = None,
        system_tracking_interval: Optional[int] = DEFAULT_SYSTEM_TRACKING_INT,
        log_system_params: bool = True,
        capture_terminal_logs: Optional[bool] = True,
        rank_zero_only: bool = True,
        entity: Optional[str] = None,
        project: Optional[str] = None,
        upload_on_close: bool = True,
        tags: Optional[Sequence[str]] = None,
        hparams_to_tags: Optional[Dict[str, str]] = (('model.pretrained_model_name_or_path', 'MN'), ('global_train_batch_size', 'GBS'), ('train_loader.dataset.local', 'DS')),
    ):
        """
        Args:
            repo (str, optional): Path or URI for Aim repo location.
            experiment_name (str, optional): The Aim experiment name.
            system_tracking_interval (int, optional): Controls how often system usage is logged.
            log_system_params (bool): Whether to log system-level metrics (CPU usage, etc.).
            capture_terminal_logs (bool, optional): Whether to capture stdout logs.
            rank_zero_only (bool): Whether to log only on the global rank zero process.
            entity (str, optional): For parity with WandB. Not strictly required by Aim.
            project (str, optional): For parity with WandB. Not strictly required by Aim.
            upload_on_close (bool): Whether to upload the Aim repo on trainer close.
            tags (Sequence[str], optional): Tags to add to the Aim run.
            hparams_to_tags (Dict[str, str], optional): Hyperparameters to add to the Aim run as tags. 
                Use dot notation to specify nested hyperparameters. e.g. {'model.pretrained_model_name_or_path':'MN', 'global_train_batch_size':'GBS'}
        """
        super().__init__()
        self.repo = repo
        self.experiment_name = experiment_name
        self.system_tracking_interval = system_tracking_interval
        self.log_system_params = log_system_params
        self.capture_terminal_logs = capture_terminal_logs

        self.rank_zero_only = rank_zero_only
        # If this rank is not zero, we won't log anything
        self._enabled = (not rank_zero_only) or (_get_global_rank() == 0)

        self.entity = entity
        self.project = project
        self.run_dir: Optional[str] = None
        self.run_url: Optional[str] = None

        self._run: Optional[Run] = None
        self._run_hash: Optional[str] = None
        self._is_in_atexit = False
        atexit.register(self._set_is_in_atexit)
        self.upload_on_close = upload_on_close
        self.tags = tags
        self.hparams_to_tags = dict(hparams_to_tags) if hparams_to_tags is not None else {}

    def _set_is_in_atexit(self):
        self._is_in_atexit = True

    @property
    def run(self) -> Run:
        """Get the underlying Aim Run instance, initializing if needed."""
        if not self._run:
            self._setup()
        return self._run

    def _add_env_var_tags(self):
        user = os.environ.get('LRG_USER')
        if user: self._run.add_tag(f"U-{user}")
        tags = os.environ.get('LRG_TAGS')
        if tags: [self._run.add_tag(t) for t in tags.split(',') if t]
    
    def _add_gpu_tag(self):
        """Add GPU type and count as tag (e.g. 'GPU-2x3090')"""
        try:
            import re, subprocess
            gpu_info = subprocess.check_output('nvidia-smi -L', shell=True).decode()
            gpu_types = re.findall(r':\s*(?:NVIDIA\s+)?(?:GeForce\s+)?(?:RTX|GTX\s+)?([A-Z0-9 \-_]+)\s*\(', gpu_info)
            if not gpu_types: return
            main_type = gpu_types[0].strip()
            count = len([t for t in gpu_types if t.strip() == main_type])
            self._run.add_tag(f"GPU-{main_type}x{count}")
        except Exception as e:
            print(f"Failed to add GPU tag: {e}")
            sys_logger.warning(f"Failed to add GPU tag: {e}")

    def _setup(self, state: Optional[State] = None):
        """Initialize the Aim Run if not already initialized."""
        if self._run is not None or not self._enabled:
            return
        
        try:
            if self._run_hash:
                self._run = Run(
                    self._run_hash,
                    repo=self.repo,
                    system_tracking_interval=self.system_tracking_interval,
                    log_system_params=self.log_system_params,
                    capture_terminal_logs=self.capture_terminal_logs,
                )
            else:
                # Provide the composer run_name if not explicitly given
                # (Aim calls this "experiment", so we can unify them)
                experiment = self.experiment_name
                if experiment is None and state is not None and state.run_name is not None:
                    experiment = state.run_name

                self._run = Run(
                    repo=self.repo,
                    experiment=experiment,
                    system_tracking_interval=self.system_tracking_interval,
                    log_system_params=self.log_system_params,
                    capture_terminal_logs=self.capture_terminal_logs,
                )
                self._run_hash = self._run.hash
            

            # If available, store or conceive a notion of "run_dir" or "run_url"
            # (Aim doesn't natively provide both. You can store custom info if desired.)
            self.run_dir = self._run.repo.path if self._run.repo else None
            # For illustration, store "entity/project" in run params if given
            if self.entity or self.project:
                self._run['meta/entity'] = self.entity
                self._run['meta/project'] = self.project

            # Optionally log initial state as hyperparameters
            if state:
                self._log_hparams(state)
            
            self._add_env_var_tags()
            self._add_gpu_tag()
            if self.tags:
                if isinstance(self.tags, str): self._run.add_tag(self.tags)
                else: [self._run.add_tag(t) for t in list(self.tags)]
        
        except Exception as e:
            sys_logger.error(f"Failed to initialize Aim run: {e}")
            raise RuntimeError(f"Aim logger initialization failed: {e}") from e

    def _log_hparams(self, state: State):
        """Log hyperparameters from the training state."""
        if not self._enabled:
            return
        try:
            default_hparams = {
                'batch_size': getattr(state.dataloader, 'batch_size', None),
                'max_duration': str(state.max_duration) if state.max_duration else None,
                'optimizer': state.optimizers[0].__class__.__name__ if state.optimizers and len(state.optimizers) > 0 else None,
            }
            if state.model: default_hparams['model_class'] = state.model.__class__.__name__
            for k, v in default_hparams.items():
                self._run.set(('state', k), v)
            state_dict = state.state_dict()
            if state_dict:
                for k, v in state_dict.items():
                    self._run.set(('state', k), v)

            # If you want to log your entire config dictionary, you can do so:
            # self._run['composer/config'] = state.get_serialized_attributes()  # Example only
            # If you want to log your entire config dictionary, you can do so:
            # self._run['composer/config'] = state.get_serialized_attributes()  # Example only
            # Or if the user's separate hyperparameter dictionary is known, do:
            # self._run['composer/hparams'] = some_dict

        except Exception as e:
            sys_logger.warning(f'Failed to log hyperparameters: {e}')

    #####################
    # LoggerDestination #
    #####################

    def init(self, state: State, logger: Logger) -> None:
        """Initialize with the training state."""
        del logger  # unused
        # If not rank zero and rank_zero_only is True,
        # then _enabled = False and we won't do anything.
        self._setup(state)

    def _get_nested(self, d: dict, key: str) -> Any:
        """Get nested dict value from dot-separated key string."""
        return d.get(key.split('.')[0]) if '.' not in key else self._get_nested(d.get(key.split('.')[0], {}), '.'.join(key.split('.')[1:]))

    def log_hyperparameters(self, hyperparameters: dict[str, Any]):
        """Log arbitrary hyperparameters to Aim."""
        sys_logger.info(f"Logging hyperparameters: {hyperparameters}")
        print(f"Logging hyperparameters: {hyperparameters}")
        for hparam_to_tag, tag_prefix in self.hparams_to_tags.items():
            hparam_value = self._get_nested(hyperparameters, hparam_to_tag)
            if hparam_value is not None:
                if (isinstance(hparam_value, str) and '/' in hparam_value and len(hparam_value.split('/')[-1]) > 1): 
                    hparam_value = hparam_value.split('/')[-1]
                self._run.add_tag(f"{tag_prefix}-{hparam_value}")

        if not self._enabled or not self._run:
            return
        # In WandB: wandb.config.update(hyperparameters)
        # In Aim, we just store them in a nested dictionary key, or flatten them:
        for k, v in hyperparameters.items():
            self._run.set(('hparams', k), v)
        # self._run['hparams'] = {k: v for k, v in hyperparameters.items()}
        # self._run['hparams3'] = str({k: v for k, v in hyperparameters.items()})
        # self._run['hparams2'] = {'test1': 'test2', 'test3': 'test4'} 
        # self._run['hparams4'] = str({k: v for k, v in hyperparameters.items()})
        sys_logger.info(f"Finished logging hyperparameters.")
        print(f"Finished logging hyperparameters.")

    def log_table(
        self,
        columns: list[str],
        rows: list[list[Any]],
        name: str = 'Table',
        step: Optional[int] = None,
    ) -> None:
        """Log tabular data. Aim does not have a 'Table' widget, but we can store it as a custom object."""
        if not self._enabled or not self._run:
            return
        table_data = {
            'columns': columns,
            'rows': rows,
        }
        # Typically you'd store as just a dictionary, or in custom namespace:
        self._run.track(table_data, name=name, step=step, context={'type': 'table'})

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to Aim."""
        if not self._enabled or not self._run:
            return
        
        try:
            # Batch metrics instead of logging individually
            metric_dict = {}
            array_metrics = {}
            
            for name, value in metrics.items():
                if value is None:
                    continue
                    
                if isinstance(value, (int, float)):
                    metric_dict[name] = value
                elif isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        metric_dict[name] = value.item()
                    else:
                        # Store array metrics separately
                        array_metrics[name] = value.detach().cpu().numpy()
                        
            # Log scalar metrics in batch
            if metric_dict:
                self._run.track(metric_dict, step=step)
                
            # Log array metrics
            for name, value in array_metrics.items():
                self._run.track(
                    value,
                    name=name,
                    step=step,
                    context={"type": "array"}
                )
                
        except Exception as e:
            sys_logger.warning(f"Failed to log metrics: {e}")

    def log_images(
        self,
        images: Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]],
        name: str = 'Images',
        step: Optional[int] = None,
        **kwargs
    ):
        """Log images, optionally with segmentation masks, to Aim.

        Aim doesn't have direct "mask overlay" or "table" features, but we can store them as separate objects.
        """
        if not self._enabled or not self._run:
            return

        # Add proper context for better organization
        context = {
            "type": "image",
            "format": "CHW" if not kwargs.get('channels_last', False) else "HWC",
            **kwargs
        }
        
        # Convert to sequence if single image
        if not isinstance(images, Sequence):
            images = [images]
        
        for idx, img in enumerate(images):
            img_data = _to_numpy_image(img, channels_last=kwargs.get('channels_last', False))
            self._run.track(
                img_data,
                name=f"{name}/{idx}" if len(images) > 1 else name,
                step=step,
                context=context
            )

    ### THIS DOES NOT WORK CURRENTLY - AIM DOESN'T NATIVELY SUPPORT ARTIFACT STORAGE SO IT REQUIRES A CUSTOM APPROACH ###
    # def upload_file(
    #     self,
    #     state: State,
    #     remote_file_name: str,
    #     file_path: pathlib.Path,
    #     *,
    #     overwrite: bool,
    # ):
    #     """Upload a file as an artifact. Aim does not have a direct artifact store feature, so we skip or do a custom approach."""
    #     if not self._enabled or not self._run:
    #         return
    #     # For demonstration, store metadata and the file path
    #     # In practice, you might copy the file into an Aim artifact store, or
    #     # simply log the path so that the user can find it in the UI.
    #     timestamp_state = state.timestamp.state_dict()
    #     metadata = {f'timestamp/{k}': v for k, v in timestamp_state.items()}
    #     data_obj = {
    #         'file_name': remote_file_name,
    #         'file_path': str(file_path),
    #         'metadata': metadata,
    #     }
    #     # THIS IS WRONG - can't use name parameter with dictionary tracking
    #     self._run.track(data_obj, name='uploaded_file', context={'type': 'artifact'})

    #     # Overwrite logic is not relevant if Aim behind the scenes cannot store the file again.
    #     # You could add a check to see if the file was already stored, etc.

    def can_upload_files(self) -> bool:
        return False

    def download_file(
        self,
        remote_file_name: str,
        destination: str,
        overwrite: bool = False,
        progress_bar: bool = True,
    ):
        """Download a file from an artifact store. Aim does not natively support this, so NotImplementedError."""
        raise NotImplementedError(
            "Aim does not provide a built-in artifact download mechanism. Provide a custom approach if needed."
        )

    def close(self, state: Optional[State] = None, logger: Optional[Logger] = None) -> None:
        """Close out the Aim run."""
        if not self._enabled:
            return
        if self._run:
            try:
                if hasattr(self._run, '_reporter') and self._run._reporter:
                    self._run._reporter.close()
                self._run.close()
            except Exception as e:
                # Use the module-level logger instead of the passed logger
                sys_logger.warning(f"Error during Aim run cleanup: {e}")
            finally:
                self._run = None
                if self.upload_on_close:
                    try:
                        upload_id = upload_repo(self.repo)
                        if upload_id:
                            sys_logger.info(f"\n{'#'*40}\n{'#'*10} AIM REMOTE UPLOAD SUCCESSFUL: {upload_id} {'#'*10}\n{'#'*40}")
                        else:
                            sys_logger.warning("\n{'!'*40}\n{'!'*10} AIM REMOTE UPLOAD FAILED. PLEASE RE-RUN Manually. {'!'*10}\n{'!'*40}")
                            raise RuntimeError("Aim repo upload failed. Please check your Aim remote server configuration.")
                    except Exception as e:
                        sys_logger.warning(f"\n{'!'*40}\n{'!'*10} AIM REMOTE UPLOAD FAILED. PLEASE RE-RUN Manually. {'!'*10}\n{'!'*40}\n{e}")

    def post_close(self) -> None:
        """Finalizes run after trainer closes. This parallels WandB usage of wandb.finish()."""
        # If we are in atexit or the run is not enabled, do nothing.
        if not self._enabled or self._is_in_atexit:
            return
        # No direct "finish" call in Aim, so nothing special to do here.
        pass


def _to_numpy_image(
    image: Union[np.ndarray, torch.Tensor],
    channels_last: bool = False,
) -> np.ndarray:
    """Utility to convert input image to a numpy array for logging in Aim."""
    if isinstance(image, torch.Tensor):
        if image.dtype in (torch.float16, torch.bfloat16):
            image = image.float()
        image = image.detach().cpu().numpy()
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Expected torch.Tensor or np.ndarray, got {type(image)}.")

    # Example: if channels are not last, transpose
    # This is not required for Aim to store it, but it emulates the W&B practice
    if not channels_last and image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))
    return image