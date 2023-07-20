gnome = import('gnome')

api_version = '0'
ggml_gobject_headers_subdir = 'ggml-gobject'

ggml_gobject_toplevel_introspectable_headers = files([
  'ggml-cached-model.h',
  'ggml-compute-graph.h',
  'ggml-compute-plan.h',
  'ggml-context.h',
  'ggml-gobject.h',
  'ggml-gpt.h',
  'ggml-hyperparameters.h',
  'ggml-language-model.h',
  'ggml-model-desc.h',
  'ggml-model.h',
  'ggml-ops.h',
  'ggml-tensor.h',
  'ggml-token-dictionary.h',
  'ggml-types.h',
])
ggml_gobject_toplevel_introspectable_sources = files([
  'ggml-cached-model.c',
  'ggml-compute-graph.c',
  'ggml-compute-plan.c',
  'ggml-context.c',
  'ggml-gobject.c',
  'ggml-gpt.c',
  'ggml-hyperparameters.c',
  'ggml-language-model.c',
  'ggml-model-desc.c',
  'ggml-model.c',
  'ggml-ops.c',
  'ggml-tensor.c',
  'ggml-token-dictionary.c',
  'ggml-types.c',
])
ggml_gobject_toplevel_internal_sources = files([
  'internal/ggml-async-queue-source.c',
  'internal/ggml-stream-internal.c',
])
ggml_gobject_toplevel_internal_headers = files([
  'internal/ggml-async-queue-source.h',
  'internal/ggml-context-internal.h',
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
  dependencies: [ glib, gobject, gio, ggml, m_dep, soup ]
)

introspection_sources = [
  ggml_gobject_toplevel_introspectable_headers,
  ggml_gobject_toplevel_introspectable_sources,
  ggml_enum_files
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
