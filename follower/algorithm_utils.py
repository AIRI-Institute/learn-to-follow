from typing import Optional, Literal

from pydantic import BaseModel


class AlgoBase(BaseModel):
    name: str = None
    num_process: int = 5
    device: str = 'cuda'
    parallel_backend: Literal[
        'multiprocessing', 'balanced_multiprocessing',
        'dask', 'balanced_dask', 'balanced_dask_gpu_backend',
        'sequential'] = 'sequential'
    seed: Optional[int] = 0
    preprocessing: Optional[str] = None
