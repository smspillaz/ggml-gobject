# /meson.build
#
# Toplevel meson build file for ggml-gobject.
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

project('ggml-gobject', ['c', 'cpp'],
        version: '0.0.0',
        license: 'LGPL2+',
        meson_version: '>= 0.55.0')

compiler = meson.get_compiler('c')
m_dep = compiler.find_library('m', required : true)
glib = dependency('glib-2.0')
gio = dependency('gio-2.0')
gio_unix = dependency('gio-unix-2.0')
gobject = dependency('gobject-2.0')
ggml = dependency('ggml')
soup = dependency('libsoup-3.0')

ggml_gobject_inc = include_directories('.')

subdir('ggml-gobject')
subdir('service')
subdir('tests')
subdir('examples')