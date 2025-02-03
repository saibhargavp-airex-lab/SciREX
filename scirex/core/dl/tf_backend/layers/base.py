# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For any clarifications or special considerations,
# please contact: contact@scirex.org
"""
    File: base.py
    Description: This module contains the base class and abstract methods 
                 for creating layers in the TensorFlow backend

    Authors:
        - Divij Ghose (divijghose@{iisc.ac.in})

    Version Info:
        - 31/01/2025: Initial version

"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
import tensorflow as tf
from ..activations import *
from ..datautils import *
from ..mathutils import *


class TensorflowLayer(ABC):
    """Base class for creating layers in the TensorFlow backend

    Provides abstract methods for creating and configuring layers in the
    TensorFlow backend. Subclasses should implement these methods.

    Example:
        >>> class MyLayer(TensorflowLayer):
        ...     def __init__(self, **kwargs):
        ...         super(MyLayer, self).__init__(**kwargs)
        ...
        ...     @abstractmethod
        ...     def create_layer(self, **kwargs):
        ...         pass
        ...
        ...     @abstractmethod
        ...     def configure_layer(self, **kwargs):
        ...         pass
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def create_layer(self, **kwargs):
        pass

    @abstractmethod
    def configure_layer(self, **kwargs):
        pass
