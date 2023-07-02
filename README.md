# ggml-gobject

[![Meson](https://github.com/smspillaz/ggml-gobject/actions/workflows/meson.yml/badge.svg)](https://github.com/smspillaz/ggml-gobject/actions/workflows/meson.yml)

A GObject binding for [GGML](https://github.com/ggerganov/ggml).

A simple "hello world" example works, eg, loading the GPT2 model weights
and doing inference on a simple sentence. However the user currently has to
do a lot of heavy lifting and there are many assumptions about memory consumption,
since internally GGML does not allocate any memory.

See [here](tests/js/testLoadGPT2.js) for an example of using the bound API from gjs.

# Building

    # install ggml first. make sure to use -DBUILD_SHARED_LIBRARY=ON
    mkdir build
    meson build
    cd build; ninja; ninja test
