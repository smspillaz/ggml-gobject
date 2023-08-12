ggml_gobject_headers_subdir = 'ggml-client'

api_version = '0'

ggml_gobject_introspectable_client_src = files([
  'ggml-client-language-model-cursor.c',
  'ggml-client-service.c',
  'ggml-client-session.c'
])
ggml_gobject_introspectable_client_headers = files([
  'ggml-client-language-model-cursor.h',
  'ggml-client-service.h',
  'ggml-client-session.h'
])
ggml_gobject_client_internal_headers = files([
  'internal/ggml-client-language-model-cursor-internal.h',
  'internal/ggml-client-session-internal.h'
])

ggml_dbus_service_src_target = gnome.gdbus_codegen(
  'ggml-service-dbus',
  sources: 'ggml-service.xml',
  interface_prefix: 'org.ggml.',
  namespace: 'GGML',
  autocleanup: 'all',
  object_manager: true,
  extra_args: ['--glib-min-required', '2.64']
)
ggml_service_gdbus_src = ggml_dbus_service_src_target[0]
ggml_service_gdbus_header = ggml_dbus_service_src_target[1]

ggml_gobject_client_sources = [
  ggml_gobject_introspectable_client_src,
  ggml_gobject_introspectable_client_headers,
  ggml_gobject_client_internal_headers,
  ggml_service_gdbus_src,
  ggml_service_gdbus_header
]

introspection_sources = [
  ggml_gobject_introspectable_client_src,
  ggml_gobject_introspectable_client_headers
]

ggml_gobject_client_lib = shared_library(
  'ggml-client',
  ggml_gobject_client_sources,
  soversion: api_version,
  install: true,
  include_directories: [ ggml_gobject_inc ],
  dependencies: [ glib, gobject, gio, gio_unix ]
)

ggml_gobject_client_gir = gnome.generate_gir(
  ggml_gobject_client_lib,
  extra_args: ['--warn-all'],
  identifier_prefix: 'GGMLClient',
  include_directories: ggml_gobject_inc,
  includes: ['GLib-2.0', 'Gio-2.0', 'GObject-2.0'],
  install: true,
  namespace: 'GGMLClient',
  nsversion: api_version,
  sources: introspection_sources,
  symbol_prefix: 'ggml_client',
  fatal_warnings: true,
  dependencies: [ ggml ]
)[0]
