/*
 * ggml-gobject/ggml-context-internal.h
 *
 * Library code for ggml-context-internal
 *
 * Copyright (C) 2023 Sam Spilsbury.
 *
 * ggml-gobject is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * ggml-gobject is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with ggml-gobject; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <ggml/ggml.h>
#include <ggml/ggml-alloc.h>
#include <glib-object.h>
#include <ggml-gobject/ggml-context.h>
#include <ggml-gobject/ggml-types.h>

G_BEGIN_DECLS

struct _GGMLContext {
  GBytes *mem_buffer;
  struct ggml_context *ctx;
  struct ggml_allocr *alloc;
  size_t ref_count;
};

G_END_DECLS