gtest_proj = subproject('gtest')
gtest_dep = gtest_proj.get_variable('gtest_dep')
gmock_dep = gtest_proj.get_variable('gmock_dep')
gmock_main_dep = gtest_proj.get_variable('gmock_main_dep')

tests_src = files([
  'load_model.cpp'
])
tests_exec = executable(
  'cpp-tests',
  tests_src,
  dependencies : [
    gtest_dep,
    gmock_dep,
    gmock_main_dep,
    glib,
    gio,
    ggml,
    soup
  ],
  link_with: [
    ggml_gobject_lib
  ],
  include_directories : [
    ggml_gobject_inc
  ]
)

test('cpp-tests', tests_exec)
