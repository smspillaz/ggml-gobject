/*
 * ggml-gobject/ggml-tensor.h
 *
 * Header file for ggml-tensor
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
#include <ggml-gobject/ggml-types.h>

G_BEGIN_DECLS

typedef struct _GGMLTensor GGMLTensor;

#define GGML_TYPE_TENSOR (ggml_tensor_get_type ())
GType ggml_tensor_get_type (void);

GGMLTensor *ggml_tensor_ref (GGMLTensor *tensor);
void ggml_tensor_unref (GGMLTensor *tensor);
size_t ggml_tensor_element_size (GGMLTensor *tensor);
size_t ggml_tensor_n_elements (GGMLTensor *tensor);
size_t ggml_tensor_block_size (GGMLTensor *tensor);
size_t ggml_tensor_n_bytes (GGMLTensor *tensor);
void ggml_tensor_set_data (GGMLTensor *tensor, char *data, size_t size);
void ggml_tensor_set_data_from_bytes (GGMLTensor *tensor, GBytes *bytes);
void ggml_tensor_set_data_from_int32_array (GGMLTensor *tensor,
                                            int32_t    *array,
                                            size_t      n_elements);
char * ggml_tensor_get_data (GGMLTensor *tensor, size_t *out_n_bytes);
GBytes * ggml_tensor_get_bytes (GGMLTensor *tensor);

void ggml_tensor_set_name (GGMLTensor *tensor,
                           const char *name);
const char * ggml_tensor_get_name (GGMLTensor *tensor);
GGMLDataType ggml_tensor_get_data_type (GGMLTensor *tensor);
int64_t *ggml_tensor_get_shape (GGMLTensor *tensor, size_t *out_n_dims);

GPtrArray * ggml_tensor_get_cgraph_children (GGMLTensor *tensor);
int64_t ggml_tensor_get_cgraph_perf_us (GGMLTensor *tensor);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLTensor, ggml_tensor_unref)

G_END_DECLS