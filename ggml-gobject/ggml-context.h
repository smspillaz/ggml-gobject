/*
 * ggml-gobject/ggml-context.h
 *
 * Header file for ggml-context
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
 * You should have received a copy of the GNU Lesser General Public License
 * along with ggml-gobject; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include <glib-object.h>
#include <ggml-gobject/ggml-tensor.h>
#include <ggml-gobject/ggml-types.h>

G_BEGIN_DECLS

typedef struct _GGMLContext GGMLContext;

#define GGML_TYPE_CONTEXT (ggml_context_get_type ())
GType ggml_context_get_type (void);

GGMLContext *ggml_context_new_from_mem_buffer (GBytes *mem_buffer);
GGMLContext *ggml_context_new (size_t memory_size);
GGMLContext *ggml_recorder_context_new (void);
GGMLContext *ggml_alloc_context_new (GBytes *mem_buffer);
GGMLContext *ggml_context_ref (GGMLContext *context);
void ggml_context_unref (GGMLContext *context);
GGMLTensor *ggml_context_new_tensor (GGMLContext  *context,
                                     GGMLDataType  data_type,
                                     int64_t      *shape,
                                     size_t        n_dims);
GGMLTensor *ggml_context_new_tensor_1d (GGMLContext *context,
                                        GGMLDataType data_type, size_t size);
GGMLTensor *ggml_context_new_tensor_2d (GGMLContext *context,
                                        GGMLDataType data_type, size_t width,
                                        size_t height);
GGMLTensor *ggml_context_new_tensor_3d (GGMLContext *context,
                                        GGMLDataType data_type, size_t width,
                                        size_t height, size_t depth);

GGMLTensor *ggml_context_new_scalar_f32 (GGMLContext *context,
                                         float value);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLContext, ggml_context_unref)

G_END_DECLS