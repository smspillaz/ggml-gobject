/*
 * ggml-gobject/ggml-ops.h
 *
 * Header file for ggml-ops
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

#include <ggml-gobject/ggml-context.h>
#include <ggml-gobject/ggml-tensor.h>

G_BEGIN_DECLS

/* Some macros here to forward declare bindings. Macros are bad
 * but these save a lot of work. */
#define GGML_DECLARE_BINARY_OP_BINDING(opname) GGMLTensor * ggml_op_ ## opname (GGMLContext *context, GGMLTensor * operand1, GGMLTensor * operand2);
#define GGML_DECLARE_UNARY_OP_BINDING(opname) GGMLTensor * ggml_op_ ## opname (GGMLContext *context, GGMLTensor * operand1);

GGML_DECLARE_BINARY_OP_BINDING (add)
GGML_DECLARE_BINARY_OP_BINDING (mul)
GGML_DECLARE_BINARY_OP_BINDING (mul_mat)
GGML_DECLARE_BINARY_OP_BINDING (cpy)
GGML_DECLARE_BINARY_OP_BINDING (get_rows)
GGML_DECLARE_BINARY_OP_BINDING (scale_inplace)
GGML_DECLARE_BINARY_OP_BINDING (repeat)
GGML_DECLARE_UNARY_OP_BINDING (soft_max_inplace)
GGML_DECLARE_UNARY_OP_BINDING (norm)
GGML_DECLARE_UNARY_OP_BINDING (transpose)
GGML_DECLARE_UNARY_OP_BINDING (gelu)

/* Some things we have to implement ourselves */
GGMLTensor * ggml_op_view_1d (GGMLContext *context, GGMLTensor *tensor, int64_t size1, size_t offset);
GGMLTensor * ggml_op_reshape_1d (GGMLContext *context, GGMLTensor *tensor, int64_t size1);
GGMLTensor * ggml_op_view_2d (GGMLContext *context, GGMLTensor *tensor, int64_t size1, int64_t size2, size_t offset);
GGMLTensor * ggml_op_reshape_2d (GGMLContext *context, GGMLTensor *tensor, int64_t size1, int64_t size2);
GGMLTensor * ggml_op_reshape_3d (GGMLContext *context, GGMLTensor *tensor, int64_t size1, int64_t size2, int64_t size3);
GGMLTensor * ggml_op_permute (GGMLContext *context, GGMLTensor *tensor, int ax1, int ax2, int ax3, int ax4);

GGMLTensor * ggml_op_diag_mask_inf_inplace (GGMLContext *context, GGMLTensor *tensor, int n_past);
GGMLTensor * ggml_op_diag_mask_zero_inplace (GGMLContext *context, GGMLTensor *tensor, int n_past);


G_END_DECLS