REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .batch_runner import BatchRunner
REGISTRY['batch'] = BatchRunner

from .parallel_info_runner import ParallelInfoRunner
REGISTRY["parallel_info"] = ParallelInfoRunner
