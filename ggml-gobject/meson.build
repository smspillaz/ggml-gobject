gnome = import('gnome')

api_version = '0'
ggml_gobject_headers_subdir = 'ggml-gobject'

ggml_gobject_toplevel_headers = files([
  'ggml.h',
])
ggml_gobject_toplevel_introspectable_headers = files([
  'ggml-gobject.h',
])
ggml_gobject_toplevel_introspectable_sources = files([
  'ggml-gobject.c',
])
ggml_enum_files = gnome.mkenums_simple('ggml-enum-types',
  sources: ggml_gobject_toplevel_introspectable_headers,
  identifier_prefix: 'GGML',
  symbol_prefix: 'ggml'
)
ggml_gobject_toplevel_sources = ggml_gobject_toplevel_introspectable_sources + ggml_enum_files

ggml_gobject_lib = shared_library(
  'ggml-gobject',
  ggml_gobject_toplevel_sources,
  soversion: api_version,
  install: true,
  include_directories: [ ggml_gobject_inc ],
  dependencies: [ glib, gobject, gio, ggml_dep, m_dep ]
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
  includes: ['GLib-2.0', 'Gio-2.0', 'GObject-2.0'],
  install: true,
  namespace: 'GGML',
  nsversion: api_version,
  sources: introspection_sources,
  symbol_prefix: 'ggml',
  fatal_warnings: true,
  dependencies: [ ggml_dep ]
)[0]