# Third-party notices

Quinlan builds on [Qwentin](https://github.com/kacper-daftcode/qwentin),
revision `8a626374912fe1e24649403423c1120ff58d8522`, by the qwentin authors.
Qwentin is licensed under the MIT license reproduced in LICENSE.
`src/quinlan/tool_parser.py` is copied from its tool parser. The native patch
contains changes to its engine and scheduler. The API layer and optional router
are adaptations of the integration developed alongside that engine.

The native sampler uses NVIDIA CUDA and CUB from the separately installed CUDA
Toolkit; those dependencies retain their own licenses. No toolkit code is vendored.
Optional conversion uses [ExLlamaV3](https://github.com/turboderp-org/exllamav3)
revision `da68d57de61781e5b74ad68698ac4caf36408d7a` under its own license.
Model weights and tokenizer files are not distributed here. Obtain them from
their publishers and observe their licenses.
