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

import multiprocessing as mp

from .logging import setup_default_logging
from .utilities import (array_split, get_data_file_path, print_progress, run_command_and_check, temporary_cd,
                        to_global_idx)


def set_default_mp_start_method(method='spawn'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method(method)
    else:
        assert mp.get_start_method() == method


set_default_mp_start_method()
