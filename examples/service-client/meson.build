client_sources = files([
  'ggml-language-model-client.c'
])

client = executable(
  'ggml-language-model-client',
  client_sources + ggml_service_gdbus_header,
  dependencies : [
    glib,
    gio,
    gio_unix,
    ggml
  ],
  link_with: [
    ggml_gobject_lib,
    ggml_gobject_client_lib
  ],
  include_directories : [
    ggml_gobject_inc
  ]
)
