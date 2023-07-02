/*
 * ggml-gobject/ggml-tensor-internal.h
 *
 * Library code for ggml-tensor-internal
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

#include <glib-object.h>
#include <ggml-gobject/ggml-context.h>
#include <ggml-gobject/ggml-types.h>
#include <ggml.h>

G_BEGIN_DECLS

struct _GGMLTensor {
  GGMLContext *owning_context;
  struct ggml_tensor *tensor;
  size_t ref_count;
};

GGMLTensor *
ggml_tensor_from_tensor (GGMLContext *context, struct ggml_tensor *base_tensor);

GGMLTensor *
ggml_tensor_new_1d (GGMLContext *context, GGMLDataType data_type, size_t size);

GGMLTensor *
ggml_tensor_new_2d (GGMLContext *context, GGMLDataType data_type, size_t width, size_t height);

GGMLTensor *
ggml_tensor_new_3d (GGMLContext *context, GGMLDataType data_type, size_t width, size_t height, size_t depth);

GGMLTensor * ggml_tensor_new_scalar_f32 (GGMLContext *context,
                                         float        value);

G_END_DECLS