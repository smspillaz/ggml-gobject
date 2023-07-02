javascript_tests = [
  'testLoadGPT2.js',
]

gjs = find_program('gjs', required: false)
jasmine = find_program('jasmine', required: false)
include_path = '@0@:@1@'.format(meson.source_root(), meson.build_root())
built_library_path = join_paths(meson.build_root(), meson.project_name())
built_js_tests_library_path = join_paths(meson.current_build_dir())
library_paths = '@0@:@1@'.format(built_library_path, built_js_tests_library_path)

tests_environment = environment()
tests_environment.set('GJS_PATH', include_path)
tests_environment.prepend('GI_TYPELIB_PATH', library_paths)
tests_environment.prepend('LD_LIBRARY_PATH', library_paths)
tests_environment.prepend('DYLD_LIBRARY_PATH', library_paths)

if jasmine.found()
  foreach test_file : javascript_tests
    test(test_file,
         gjs,
         args: [
           jasmine.full_path(),
           '--verbose',
           join_paths(meson.current_source_dir(), test_file)
         ],
         env: tests_environment,
         timeout: 0)
  endforeach
endif
