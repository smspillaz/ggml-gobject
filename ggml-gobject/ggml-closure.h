/*
 * ggml-gobject/internal/ggml-closure.h
 *
 * Library code for ggml-closure
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

#pragma once

#include <glib-object.h>

G_BEGIN_DECLS

typedef struct _GGMLClosure GGMLClosure;

#define GGML_TYPE_CLOSURE (ggml_closure_get_type ())
GType ggml_closure_get_type (void);

GGMLClosure * ggml_closure_new (GCallback      callback,
                                gpointer       user_data,
                                GDestroyNotify user_data_destroy);
GGMLClosure * ggml_closure_ref (GGMLClosure *closure);
void ggml_closure_unref (GGMLClosure *closure);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLClosure, ggml_closure_unref);

G_END_DECLS
