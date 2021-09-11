from dask import delayed, bag
from shutil import rmtree
from glob import glob
import os
from typing import List
import toolz as tz
from typing import Sequence, Union, Any, Tuple
from pathlib import Path
import fsspec


@delayed
def _rmtree_after_delete_files(path: str, dependency: Any):
    rmtree(path)


def rmtree_parallel(
    path: Union[str, Path], branch_depth: int = 1, compute: bool = True
):
    branches = glob(os.path.join(path, *("*",) * branch_depth))
    deleter = os.remove
    files = list_files_parallel(branches)
    deleted_files = bag.from_sequence(files).map(deleter)
    result = _rmtree_after_delete_files(path, dependency=deleted_files)

    if compute:
        return result.compute(scheduler="threads")
    else:
        return result


def list_files(paths: Union[Sequence[Union[str, Path]], str, Path], followlinks=False):
    if isinstance(paths, str) or isinstance(paths, Path):
        if os.path.isdir(paths):
            return list(
                tz.concat(
                    (os.path.join(dp, f) for f in fn)
                    for dp, dn, fn in os.walk(paths, followlinks=followlinks)
                )
            )
        elif os.path.isfile(paths):
            return [paths]
        else:
            raise ValueError(f"Input argument {paths} is not a path or a directory")

    elif isinstance(paths, Sequence):
        sortd = sorted(paths, key=os.path.isdir)
        files, dirs = tuple(tz.partitionby(os.path.isdir, sortd))
        return list(tz.concatv(files, *tz.map(list_files, dirs)))


def list_files_parallel(
    paths: Union[Sequence[Union[str, Path]], str, Path],
    followlinks=False,
    compute: bool = True,
):
    result = []
    delf = delayed(lambda p: list_files(p, followlinks=followlinks))

    if isinstance(paths, str) or isinstance(paths, Path):
        result = bag.from_delayed([delf(paths)])
    elif isinstance(paths, Sequence):
        result = bag.from_delayed([delf(p) for p in paths])
    else:
        raise TypeError(f"Input must be a string or a sequence, not {type(paths)}")

    if compute:
        return result.compute(scheduler="threads")
    else:
        return result


def split_path_at_suffix(urlpath: str, suffixes: Sequence[str]) -> Tuple[str, str, str]:
    """
    Given a string representing a path on a filesystem and a collection of suffixes, return
    the path split at the last instance of any element of the path containing one of the
    suffixes, as well as the suffix. If the last element of the path bears a suffix, return the path,
    the empty string, and the suffix.
    """
    protocol: str
    subpath: str
    protocol, subpath = fsspec.core.split_protocol(urlpath)
    if protocol is None:
        separator = os.path.sep
    else:
        separator = "/"
    parts = Path(subpath).parts
    suffixed = [Path(part).suffix in suffixes for part in parts]

    if not any(suffixed):
        raise ValueError("No suffixed path elements found")

    index = [idx for idx, val in enumerate(suffixed) if val][-1]
    if index == (len(parts) - 1):
        pre, post = subpath, ""
    else:
        pre, post = (
            separator.join([p.strip(separator) for p in parts[: index + 1]]),
            separator.join([p.strip(separator) for p in parts[index + 1 :]]),
        )

    suffix = Path(pre).suffix
    if protocol:
        pre = f"{protocol}://{pre}"
    return pre, post, suffix
