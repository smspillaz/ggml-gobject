gnome = import('gnome')

subdir('dbus')

api_version = '0'
ggml_gobject_headers_subdir = 'ggml-gobject'

ggml_gobject_toplevel_introspectable_headers = files([
  'ggml-argmax-language-model-sampler.h',
  'ggml-cached-model.h',
  'ggml-closure.h',
  'ggml-compute-graph.h',
  'ggml-compute-plan.h',
  'ggml-context.h',
  'ggml-execution-memory.h',
  'ggml-functional-language-model-sampler.h',
  'ggml-gobject.h',
  'ggml-gpt.h',
  'ggml-hyperparameters.h',
  'ggml-language-model.h',
  'ggml-language-model-sampler.h',
  'ggml-model-config.h',
  'ggml-model-desc.h',
  'ggml-model.h',
  'ggml-ops.h',
  'ggml-quantize.h',
  'ggml-tensor.h',
  'ggml-token-dictionary.h',
  'ggml-top-k-top-p-language-model-sampler.h',
  'ggml-types.h',
])
ggml_gobject_toplevel_introspectable_sources = files([
  'ggml-argmax-language-model-sampler.c',
  'ggml-cached-model.c',
  'ggml-closure.c',
  'ggml-compute-graph.c',
  'ggml-compute-plan.c',
  'ggml-context.c',
  'ggml-execution-memory.c',
  'ggml-functional-language-model-sampler.c',
  'ggml-gobject.c',
  'ggml-gpt.c',
  'ggml-hyperparameters.c',
  'ggml-language-model.c',
  'ggml-language-model-sampler.c',
  'ggml-model-config.c',
  'ggml-model-desc.c',
  'ggml-model.c',
  'ggml-ops.c',
  'ggml-quantize.c',
  'ggml-tensor.c',
  'ggml-token-dictionary.c',
  'ggml-top-k-top-p-language-model-sampler.c',
  'ggml-types.c',
])
ggml_gobject_toplevel_internal_sources = files([
  'internal/ggml-async-queue-source.c',
  'internal/ggml-progress-istream.c',
  'internal/ggml-stream-internal.c',
])
ggml_gobject_toplevel_internal_headers = files([
  'internal/ggml-async-queue-source.h',
  'internal/ggml-closure-internal.h',
  'internal/ggml-context-internal.h',
  'internal/ggml-progress-istream.h',
  'internal/ggml-stream-internal.h',
  'internal/ggml-tensor-internal.h',
])
ggml_enum_files = gnome.mkenums_simple('ggml-enum-types',
  sources: ggml_gobject_toplevel_introspectable_headers,
  identifier_prefix: 'GGML',
  symbol_prefix: 'ggml'
)

ggml_gobject_toplevel_sources = ggml_gobject_toplevel_introspectable_sources + ggml_gobject_toplevel_internal_sources + ggml_enum_files

ggml_gobject_lib = shared_library(
  'ggml-gobject',
  ggml_gobject_toplevel_sources,
  soversion: api_version,
  install: true,
  include_directories: [ ggml_gobject_inc ],
  dependencies: [ glib, gobject, gio, gio_unix, ggml, m_dep, soup ]
)

introspection_sources = [
  ggml_gobject_toplevel_introspectable_headers,
  ggml_gobject_toplevel_introspectable_sources,
  ggml_enum_files,
  ggml_service_gdbus_src
]

ggml_gobject_gir = gnome.generate_gir(
  ggml_gobject_lib,
  extra_args: ['--warn-all', '-U', '__ARM_NEON'],
  identifier_prefix: 'GGML',
  include_directories: ggml_gobject_inc,
  includes: ['GLib-2.0', 'Gio-2.0', 'GObject-2.0', 'Soup-3.0'],
  install: true,
  namespace: 'GGML',
  nsversion: api_version,
  sources: introspection_sources,
  symbol_prefix: 'ggml',
  fatal_warnings: true,
  dependencies: [ ggml ]
)[0]
