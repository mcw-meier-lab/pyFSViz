# Licensed under package MIT License
# Copyright (c) 2025-2026 mcw-meier-lab

# HTML reporting derived from nipreps MRIQC setup.
# Changes involve html content: FreeSurfer-related images and plots.
# Intended to keep the look and feel of nipreps MRIQC-style reports, but
# for FreeSurfer data.

# ORIGINAL WORK'S ATTRIBUTION NOTICE
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#


"""Utility class for generating a config file from a jinja template.

https://github.com/oesteban/endofday/blob/f2e79c625d648ef45b08cc1f11fd0bd84342d604/endofday/core/template.py.

Along with other report-related functions.

"""

from pathlib import Path

import jinja2


class Template:
    """Simplified jinja2 template class from oesteban."""

    def __init__(self, template_str: str) -> None:
        # Load relative to the template's parent directory so absolute paths work
        # on Windows (FileSystemLoader("/") only works on Unix).
        template_path = Path(template_str).resolve()
        self._template_name = template_path.name
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=str(template_path.parent)),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=True,
        )

    def compile(self, configs: dict) -> str:
        """Generate a string with the replacements."""
        template = self._env.get_template(self._template_name)
        return template.render(configs)

    def generate_conf(self, configs: dict, path: str) -> None:
        """Save the outcome after replacement on the template to file."""
        output = self.compile(configs)
        with open(path, "w+", encoding="utf-8") as output_file:
            output_file.write(output)
