# ggml-gobject

[![Meson](https://github.com/smspillaz/ggml-gobject/actions/workflows/meson.yml/badge.svg)](https://github.com/smspillaz/ggml-gobject/actions/workflows/meson.yml)

A GObject binding for [GGML](https://github.com/ggerganov/ggml).

A simple "hello world" example works, eg, loading the GPT2 model weights
and doing inference on a simple sentence.

The tests in [JavaScript](tests/js/testLoadGPT2.js) and [C++](tests/c/load_model.cpp) show examples of how
the API can be used from different languages.

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

# Usage

There are two ways to use the library - directly through its API, or through `ggml-service`, a D-Bus service which exposes the main
functionality of the library. The service is sophisticated enough that it supports things like resumable completions and stateful
sessions, so if there's no need to do anything fancy like specifying your own model architecture, using the service means that you
can avoid a direct dependency.

Using the service also has the advantage that in the future, the service will try to manage sessions and memory. Language model inference
can be very compute and memory intensive, so multiple applications trying to do it at the same time as a bad idea.

To use the service, it is recommended to use the `ggml-client` library, as opposed to trying to interface over D-Bus directly. The reason is
that the D-Bus flow requires opening a private session with the service library, which in turn requires opening unix pipes etc.
