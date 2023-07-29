# ggml-gobject

[![Meson](https://github.com/smspillaz/ggml-gobject/actions/workflows/meson.yml/badge.svg)](https://github.com/smspillaz/ggml-gobject/actions/workflows/meson.yml)

A GObject binding for [GGML](https://github.com/ggerganov/ggml).

A simple "hello world" example works, eg, loading the GPT2 model weights
and doing inference on a simple sentence. However the user currently has to
do a lot of heavy lifting and there are many assumptions about memory consumption,
since internally GGML does not allocate any memory.

See [here](tests/js/testLoadGPT2.js) for an example of using the bound API from gjs.

At present the API and ABI are *not* stable, and may change between commmits.

# Building

    # install ggml first. make sure to use -DBUILD_SHARED_LIBRARY=ON
    mkdir build
    meson build
    cd build; ninja; ninja test

# Example

[See a video of an example app here](https://sspilsbury-com-images.s3.amazonaws.com/llm_writer_recording.mov)

You can run the example app like:

    GI_TYPELIB_PATH=/usr/local/lib64/girepository-1.0:$GI_TYPELIB_PATH LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib:$LD_LIBRARY_PATH /usr/local/bin/org.ggml-gobject.LLMWriter.Application 
