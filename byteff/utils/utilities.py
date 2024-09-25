# -----  ByteFF: ByteDance Force Field -----
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import errno
import logging
import math
import os
import subprocess
from contextlib import contextmanager
from pathlib import PosixPath
from tempfile import TemporaryDirectory
from typing import Generator, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def print_progress(finished: int, total: int, description: str = ''):
    nparts = int(finished / total * 60)
    string = f"{description} {finished}/{total} "
    string += '|' + '█' * nparts + ' ' * (60 - nparts) + '|'
    print(string, end='\r' if finished < total else '\n')


def batch_to_single(batched_data: torch.Tensor, counts: torch.LongTensor, index: int):
    assert counts.dim() == 1 and counts.sum() == batched_data.shape[0]
    assert isinstance(index, int) and index >= 0
    csum = torch.cumsum(counts, dim=0)
    if index == 0:
        return batched_data[:csum[0]]
    elif index > 0:
        return batched_data[csum[index - 1]:csum[index]]


@contextmanager
def temporary_cd(directory_path: Optional[str] = None) -> Generator[None, None, None]:
    """Temporarily move the current working directory to the path
    specified. If no path is given, a temporary directory will be
    created, moved into, and then destroyed when the context manager
    is closed.

    Parameters
    ----------
    directory_path: str, optional

    Returns
    -------

    """
    if isinstance(directory_path, PosixPath):
        directory_path = directory_path.as_posix()

    if directory_path is not None and len(directory_path) == 0:
        yield
        return

    old_directory = os.getcwd()

    try:

        if directory_path is None:

            with TemporaryDirectory() as new_directory:
                os.chdir(new_directory)
                yield

        else:

            os.makedirs(directory_path, exist_ok=True)
            os.chdir(directory_path)
            yield

    finally:
        os.chdir(old_directory)


def get_data_file_path(relative_path: str, package_name: str) -> str:
    """Get the full path to one of the files in the data directory.

    If no file is found at `relative_path`, a second attempt will be made
    with `data/` preprended. If no files exist at either path, a FileNotFoundError
    is raised.

    Parameters
    ----------
    relative_path : str
        The relative path of the file to load.
    package_name : str
        The name of the package in which a file is to be loaded, i.e.

    Returns
    -------
        The absolute path to the file.

    Raises
    ------
    FileNotFoundError
    """
    from importlib.resources import files

    file_path = files(package_name) / relative_path

    if not file_path.is_file():
        try_path = files(package_name) / f"data/{relative_path}"
        if try_path.is_file():
            file_path = try_path
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)

    return file_path.as_posix()  # type: ignore


def run_command_and_check(cmd: str,
                          *,
                          allow_error: bool = False,
                          separate_stderr: bool = True,
                          env: dict = None,
                          redirect_stdout=True,
                          user_input: str = None,
                          timeout: float = None) -> tuple[int, str, str]:

    cmdenv = dict()
    cmdenv.update(**os.environ)
    if env is not None:
        cmdenv.update(**env)

    logger.debug(f'running cmd {cmd} with extra env {env}')
    stdout = subprocess.PIPE if redirect_stdout else None
    stderr = subprocess.PIPE if separate_stderr else subprocess.STDOUT
    stderr = stderr if redirect_stdout else None
    user_input = user_input.encode() if isinstance(user_input, str) else None
    result = subprocess.run(cmd,
                            stdout=stdout,
                            stderr=stderr,
                            env=cmdenv,
                            shell=True,
                            check=False,
                            input=user_input,
                            timeout=timeout)

    stdout = result.stdout.decode('utf-8') if result.stdout is not None else None
    stderr = result.stderr.decode('utf-8') if result.stderr is not None else None

    if result.returncode != 0 and not allow_error:
        logger.error(f'cmd {cmd}')
        logger.error(f'return code {result.returncode}')
        logger.error(f'stdout {stdout}')
        logger.error(f'stderr {stderr}')
        raise RuntimeError(f'fail to run "{cmd}". ')

    return result.returncode, stdout, stderr


def array_split(array: list, num_parts: int) -> list[list]:
    """
    Split an array into num_parts subarrays of roughly equal size.

    Args:
        array: An array to be split.
        num_parts: The number of subarrays to split the array into.

    Returns:
        A list of lists, where each sublist contains a roughly equal portion of the original array.
        The number of sublists equals num_parts, unless the length of the array is not evenly divisible by num_parts.
    """
    num_elements_per_part = math.ceil(len(array) / num_parts)
    vacancy = num_parts * num_elements_per_part - len(array)
    subarrays = []
    start_index = 0

    for i in range(num_parts):
        end_index = start_index + num_elements_per_part - 1 if i >= num_parts - vacancy else start_index + num_elements_per_part
        subarrays.append(array[start_index:end_index])
        start_index = end_index

    return subarrays


def to_global_idx(local_idx: Tensor, shifts: Tensor, counts: Tensor):
    """ get the global_idx from local_idx after batch
        shifts is the number to shift for each data, 
        counts is the number of terms in each molecule
    """
    local_idx = local_idx.long()
    shifts = shifts.long()
    counts = counts.long()
    nedges_cumsum = torch.cumsum(shifts, 0)  # [batch_size]
    nedges_cumsum = torch.concat((torch.tensor([0], device=shifts.device, dtype=shifts.dtype), nedges_cumsum[:-1]),
                                 dim=0)  # [batch_size]
    size = (-1,) + (1,) * (local_idx.dim() - 1)
    global_idx = local_idx + torch.repeat_interleave(nedges_cumsum, counts).view(size)  # [nterm, ...]
    return global_idx
