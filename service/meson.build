gnome = import('gnome')

server_sources = files([
  'ggml-language-model-service.c'
])

server = executable(
  'ggml-language-model-service',
  server_sources,
  dependencies : [
    glib,
    gio,
    gio_unix,
    ggml
  ],
  link_with: [
    ggml_gobject_lib
  ],
  include_directories : [
    ggml_gobject_inc
  ]
)
