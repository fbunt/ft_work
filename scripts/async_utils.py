from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import concurrent.futures as cfutures
import itertools


class AsyncJobError(Exception):
    pass


MULTI_PROCESS = "multiprocess"
MULTI_THREAD = "multithread"


# TODO: add generic typing
class AsyncJob:
    """A class representing a job to be run in the background using
    `run_async_jobs`.
    """

    # Used to get unique id within current process.
    next_job_id = itertools.count()

    def __init__(self, func: Callable, *args: Any):
        self._id = next(AsyncJob.next_job_id)
        self.func = func
        self.args = args
        self.result = None

    def run(self) -> Any:
        self.result = self.func(*self.args)
        return self.result


def _run_jobs(*jobs: AsyncJob):
    return {job._id: job.run() for job in jobs}


def run_async_jobs(
    jobs: Sequence[AsyncJob],
    async_type=MULTI_THREAD,
    max_workers=None,
    chunk_size=1,
) -> List[Any]:
    """Run a set of `AsyncJob`s asynchronously and return the results.

    This is a high-level helper function.

    Arguments:
        jobs: List of jobs to be run
        async_type: Indicates whether threads or full processes should be used.
        max_workers: The max number of background threads/processes to run at
                once. If None, the backend default is used.
        chunk_size: The number of jobs to run on each background thread/process
    """
    if not jobs:
        raise ValueError("No jobs provided")
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero")
    final_ids = [j._id for j in jobs]
    if len(final_ids) != len(set(final_ids)):
        # This should never happen becuase the job class uses a class-static id
        # generator in the constructor. If this condition turns out to be
        # possible, the id generation will need to be reworked.
        raise AsyncJobError("Jobs cannot have the same ID")

    # Break jobs into chunks
    chunks = [
        jobs[i : i + chunk_size] for i in range(0, len(jobs), chunk_size)
    ]
    chunk_ids = list(range(len(chunks)))
    results = run_async(
        _run_jobs,
        chunk_ids,
        chunks,
        async_type=async_type,
        max_workers=max_workers,
        error_on_fail=True,
    )
    out = {}
    for k in results:
        out.update(results[k])
    # Return results in original order
    return [out[i] for i in final_ids]


def run_async(
    func: Callable,
    job_ids: Iterable[Hashable],
    job_args: Optional[Iterable[Iterable[Any]]],
    async_type: str = MULTI_THREAD,
    max_workers: Optional[int] = None,
    error_on_fail: bool = True,
) -> Union[
    Dict[Hashable, Any], Tuple[Dict[Hashable, Any], Dict[Hashable, Exception]]
]:
    """Run a set of jobs asynchronously and return the results in a dict with
    IDs as keys.

    Arguments:
        func: The callable to perform the work.
        job_ids: An iterable specifying the job IDs.
        job_args: None or an iterable giving the args for each worker call.
        async_type: Indicates whether threads or full processes should be used.
        max_workers: The max number of jobs to run at once. If None, the
                backend default is used.
        error_on_fail: If True, the exception that caused a job to fail is
                raised again in the main process. If False, errors are returned
                in a second dict using the failed IDs as keys.
    """
    if not (isinstance(max_workers, int) or max_workers is None):
        raise TypeError("Max worker count must be an int or None")
    if max_workers is not None and max_workers <= 0:
        raise ValueError("Max worker count must be greater than zero")

    ex_type = None
    if async_type == MULTI_PROCESS:
        ex_type = cfutures.ProcessPoolExecutor
    elif async_type == MULTI_THREAD:
        ex_type = cfutures.ThreadPoolExecutor
    else:
        raise AsyncJobError(f"Invalid async job type: '{async_type}'")

    # Use an empty list generator if no job args were given
    job_args = job_args or itertools.repeat(())
    results = {}
    fails = {}
    with ex_type(max_workers=max_workers) as ex:
        futures_to_ids = {
            ex.submit(func, *args): id_
            for i, (id_, args) in enumerate(zip(job_ids, job_args))
        }
        remaining_futures = set(futures_to_ids)
        try:
            for w in cfutures.as_completed(futures_to_ids):
                remaining_futures.remove(w)
                id_ = futures_to_ids[w]
                try:
                    results[id_] = w.result()
                except Exception as e:
                    if error_on_fail:
                        raise e
                    fails[id_] = e
        finally:
            for rf in remaining_futures:
                rf.cancel()
    if error_on_fail:
        return results
    return results, fails
