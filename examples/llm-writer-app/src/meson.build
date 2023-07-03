# /examples/llm-writer-app/src/meson.build
#
# Meson build file for ggml-gobject llm-writer-app src.
#
# Copyright (C) 2023 Sam Spilsbury
#
# ggml-gobject is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
#
# ggml-gobject is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with ggml-gobject; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

gnome = import('gnome')
pkgdatadir = join_paths(get_option('datadir'), 'org.ggml_gobject.LLMWriter.Application')
gjs = find_program('gjs', 'gjs-console')

gnome.compile_resources(
  'org.ggml-gobject.LLMWriter.Application.src',
  'org.ggml-gobject.LLMWriter.Application.src.gresource.xml',
  gresource_bundle: true,
  install: true,
  install_dir: pkgdatadir
)

subst = configuration_data()
subst.set('GJS', gjs.path())
subst.set('PACKAGE_VERSION', meson.project_version())
subst.set('prefix', get_option('prefix'))
subst.set('libdir', get_option('libdir'))
subst.set('pkgdatadir', pkgdatadir)
configure_file(
  input: 'org.ggml-gobject.LLMWriter.Application.in',
  output: 'org.ggml-gobject.LLMWriter.Application',
  configuration: subst,
  install_dir: get_option('bindir')
)

run_target('example-llm-writer',
           command: join_paths(meson.build_root(),
                               'examples',
                               'llm-writer-app',
                               'src',
                               'org.ggml-gobject.LLMWriter.Application'))