/*
 * ggml-gobject/ggml-ops.c
 *
 * Library code for ggml-ops
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

#include <ggml-gobject/ggml-ops.h>
#include <ggml-gobject/internal/ggml-context-internal.h>
#include <ggml-gobject/internal/ggml-tensor-internal.h>

/**
 * ggml_op_view_1d:
 * @context: (transfer none): A #GGMLContext
 * @tensor: (transfer none): A #GGMLTensor
 * @size1: The size to view as
 * @offset: Number of elements into the original data to offset into
 *
 * Returns: (transfer full): A new #GGMLTensor, viewing the original data
 */
GGMLTensor *
ggml_op_view_1d (GGMLContext *context,
                 GGMLTensor  *tensor,
                 int64_t      size1,
                 size_t       offset)
{
  return ggml_tensor_from_tensor (context,
                                  ggml_view_1d (context->ctx,
                                                tensor->tensor,
                                                size1,
                                                offset * ggml_tensor_element_size (tensor)));
}

/**
 * ggml_op_reshape_1d:
 * @context: (transfer none): A #GGMLContext
 * @tensor: (transfer none): A #GGMLTensor
 * @size1: The size to reshape
 *
 * Returns: (transfer full): A new #GGMLTensor, reshaping the original data and copying
 */
GGMLTensor *
ggml_op_reshape_1d (GGMLContext *context,
                    GGMLTensor  *tensor,
                    int64_t      size)
{
  return ggml_tensor_from_tensor (context,
                                  ggml_reshape_1d (context->ctx,
                                                   tensor->tensor,
                                                   size));
}

/**
 * ggml_op_view_2d:
 * @context: (transfer none): A #GGMLContext
 * @tensor: (transfer none): A #GGMLTensor
 * @size1: The size on the first dimension
 * @size2: The size on the second dimension
 * @offset: Number of elements into the original data to offset into
 *
 * Returns: (transfer full): A new #GGMLTensor, viewing the original data
 */
GGMLTensor *
ggml_op_view_2d (GGMLContext *context,
                 GGMLTensor  *tensor,
                 int64_t      size1,
                 int64_t      size2,
                 size_t       offset)
{
  return ggml_tensor_from_tensor (context,
                                  ggml_view_2d (context->ctx,
                                                tensor->tensor,
                                                size1,
                                                size2,
                                                tensor->tensor->nb[1],
                                                offset * ggml_tensor_element_size (tensor)));
}

/**
 * ggml_op_reshape_2d:
 * @context: (transfer none): A #GGMLContext
 * @tensor: (transfer none): A #GGMLTensor
 * @size1: The size on the first dimension to reshape
 * @size2: The size on the second dimension to reshape
 *
 * Returns: (transfer full): A new #GGMLTensor, reshaping the original data and copying
 */
GGMLTensor *
ggml_op_reshape_2d (GGMLContext *context,
                    GGMLTensor  *tensor,
                    int64_t      size1,
                    int64_t      size2)
{
  return ggml_tensor_from_tensor (context,
                                  ggml_reshape_2d (context->ctx,
                                                   tensor->tensor,
                                                   size1,
                                                   size2));
}

/**
 * ggml_op_reshape_3d:
 * @context: (transfer none): A #GGMLContext
 * @tensor: (transfer none): A #GGMLTensor
 * @size1: The size on the first dimension to reshape
 * @size2: The size on the second dimension to reshape
 * @size3: The size on the third dimension to reshape
 *
 * Returns: (transfer full): A new #GGMLTensor, reshaping the original data and copying
 */
GGMLTensor *
ggml_op_reshape_3d (GGMLContext *context,
                    GGMLTensor  *tensor,
                    int64_t      size1,
                    int64_t      size2,
                    int64_t      size3)
{
  return ggml_tensor_from_tensor (context,
                                  ggml_reshape_3d (context->ctx,
                                                   tensor->tensor,
                                                   size1,
                                                   size2,
                                                   size3));
}

/**
 * ggml_op_permute:
 * @context: (transfer none): A #GGMLContext
 * @tensor: (transfer none): A #GGMLTensor
 * @ax1: New position of axis 1
 * @ax2: New position of axis 2
 * @ax3: New position of axis 3
 * @ax4: New position of axis 4
 *
 * Permutes the axis order of the tensor, so that the elements are sub-ordered
 * according to the permutation given in the arguments. You can think of
 * this as a sort of n-dimensional transpose or swapping one axis for another,
 * eg, permute(2, 1, 3, 4) swaps the first two axes (transpose)
 *
 * Returns: (transfer full): A new #GGMLTensor, with the permuted axes
 */
GGMLTensor *
ggml_op_permute (GGMLContext *context,
                 GGMLTensor  *tensor,
                 int          ax1,
                 int          ax2,
                 int          ax3,
                 int          ax4)
{
  return ggml_tensor_from_tensor (context,
                                  ggml_permute (context->ctx,
                                                tensor->tensor,
                                                ax1,
                                                ax2,
                                                ax3,
                                                ax4));
}

/**
 * ggml_op_diag_mask_inf_inplace:
 * @context: A #GGMLContext
 * @tensor: A #GGMLTensor
 * @n_past: Number of past
 *
 * Causally mask the 2D input tensor with inf values
 *
 * Returns: (transfer full): A new #GGMLTensor
 */
GGMLTensor *
ggml_op_diag_mask_inf_inplace (GGMLContext *context,
                               GGMLTensor  *tensor,
                               int          n_past)
{
  return ggml_tensor_from_tensor (context,
                                  ggml_diag_mask_inf_inplace (context->ctx,
                                                              tensor->tensor,
                                                              n_past));
}

/**
 * ggml_op_diag_mask_zero_inplace:
 * @context: A #GGMLContext
 * @tensor: A #GGMLTensor
 * @n_past: Number of past
 *
 * Causally mask the 2D input tensor with zero values
 *
 * Returns: (transfer full): A new #GGMLTensor
 */
GGMLTensor *
ggml_op_diag_mask_zero_inplace (GGMLContext *context,
                                GGMLTensor  *tensor,
                                int          n_past)
{
  return ggml_tensor_from_tensor (context,
                                  ggml_diag_mask_zero_inplace (context->ctx,
                                                               tensor->tensor,
                                                               n_past));
}

/**
 * ggml_op_norm:
 * @context: A #GGMLContext
 * @tensor: A #GGMLTensor
 * @eps: An epsilon value to add prior to normalization
 *
 * Normalize @tensor
 *
 * Returns: (transfer full): A new #GGMLTensor with the result
 */
GGMLTensor *
ggml_op_norm (GGMLContext *context,
              GGMLTensor  *tensor,
              float        eps)
{
  return ggml_tensor_from_tensor (context,
                                  ggml_norm (context->ctx,
                                             tensor->tensor,
                                             eps));
}

#define GGML_DEFINE_BINARY_OP_BINDING(opname) \
  GGMLTensor * \
  ggml_op_ ## opname (GGMLContext *context, \
                      GGMLTensor  *operand1, \
                      GGMLTensor  *operand2) \
  { \
    struct ggml_tensor *result = ggml_ ## opname (context->ctx, operand1->tensor, operand2->tensor); \
    GGMLTensor *tensor = ggml_tensor_from_tensor (context, result); \
    ggml_tensor_set_name (tensor, #opname); \
    return tensor; \
  }

#define GGML_DEFINE_UNARY_OP_BINDING(opname) \
  GGMLTensor * \
  ggml_op_ ## opname (GGMLContext *context, \
                      GGMLTensor  *operand1) \
  { \
    struct ggml_tensor *result = ggml_ ## opname (context->ctx, operand1->tensor); \
    GGMLTensor *tensor = ggml_tensor_from_tensor (context, result); \
    ggml_tensor_set_name (tensor, #opname); \
    return tensor; \
  }

/**
 * ggml_op_add:
 * @context: A #GGMLContext
 * @operand1: A #GGMLTensor
 * @operand2: A #GGMLTensor
 *
 * Elementwise add two tensors
 *
 * Returns: (transfer full): A new #GGMLTensor with the result
 */
GGML_DEFINE_BINARY_OP_BINDING (add)

/**
 * ggml_op_mul:
 * @context: A #GGMLContext
 * @operand1: A #GGMLTensor
 * @operand2: A #GGMLTensor
 *
 * Elementwise multiply two tensors
 *
 * Returns: (transfer full): A new #GGMLTensor with the result
 */
GGML_DEFINE_BINARY_OP_BINDING (mul)

/**
 * ggml_op_mul_mat:
 * @context: A #GGMLContext
 * @operand1: A #GGMLTensor
 * @operand2: A #GGMLTensor
 *
 * Multiply two matrix tensors. Requires that for tensors of shape (M, N), (P, Q)
 * that N == P and result will have shape M, Q
 *
 * Returns: (transfer full): A new #GGMLTensor with the result
 */
GGML_DEFINE_BINARY_OP_BINDING (mul_mat)

/**
 * ggml_op_cpy:
 * @context: A #GGMLContext
 * @operand1: A #GGMLTensor
 * @operand2: A #GGMLTensor
 *
 * Copies @operand2 to @operand1
 *
 * Returns: (transfer full): A new #GGMLTensor with the result
 */
GGML_DEFINE_BINARY_OP_BINDING (cpy)

/**
 * ggml_op_get_rows:
 * @context: A #GGMLContext
 * @operand1: A #GGMLTensor
 * @operand2: A #GGMLTensor
 *
 * Fetches rows from operand2 based on the indices in operand1. Sort of like
 * an embedding.
 *
 * Returns: (transfer full): A new #GGMLTensor with the result
 */
GGML_DEFINE_BINARY_OP_BINDING (get_rows)

/**
 * ggml_op_scale_inplace:
 * @context: A #GGMLContext
 * @operand1: A #GGMLTensor
 * @operand2: A #GGMLTensor
 *
 * Scales @operand1 by @operand2 (a scalar)
 *
 * Returns: (transfer full): A new #GGMLTensor with the result
 */
GGML_DEFINE_BINARY_OP_BINDING (scale_inplace)

/**
 * ggml_op_repeat:
 * @context: A #GGMLContext
 * @operand1: A #GGMLTensor
 * @operand2: A #GGMLTensor
 *
 * Repeats (broadcasts) @operand1 to match the shape of @operand2
 *
 * Returns: (transfer full): A new #GGMLTensor with the result
 */
GGML_DEFINE_BINARY_OP_BINDING (repeat)

/**
 * ggml_op_soft_max_inplace:
 * @context: A #GGMLContext
 * @operand1: A #GGMLTensor
 *
 * Computes softmax over @operand1
 *
 * Returns: (transfer full): A new #GGMLTensor with the result
 */
GGML_DEFINE_UNARY_OP_BINDING (soft_max_inplace)

/**
 * ggml_op_transpose:
 * @context: A #GGMLContext
 * @operand1: A #GGMLTensor
 *
 * Transposes @operand1
 *
 * Returns: (transfer full): A new #GGMLTensor with the result
 */
GGML_DEFINE_UNARY_OP_BINDING (transpose)

/**
 * ggml_op_gelu:
 * @context: A #GGMLContext
 * @operand1: A #GGMLTensor
 *
 * Does the GELU (Gaussian Error Linear Unit) activation
 * on @operand1
 *
 * Returns: (transfer full): A new #GGMLTensor with the result
 */
GGML_DEFINE_UNARY_OP_BINDING (gelu)