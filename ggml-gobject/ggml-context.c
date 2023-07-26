/*
 * ggml-gobject/ggml-context.c
 *
 * Library code for ggml-context
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

#include <ggml-gobject/ggml-context.h>
#include <ggml-gobject/internal/ggml-context-internal.h>
#include <ggml-gobject/internal/ggml-tensor-internal.h>

/**
 * ggml_context_new_from_mem_buffer:
 * @mem_buffer: A #GBytes with a memory pool for this context
 *
 * Creates a new #GGMLContext and memory pool from which #GGMLTensor
 * objects can be allocated. The @mem_buffer parameter's size is important -
 * you need to properly estimate it upfront, for example by using
 * what you know about the model architecture to be allocated, or
 * how many elements you will have in the input etc.
 *
 * Returns: (transfer full): A new #GGMLContext
 */
GGMLContext *
ggml_context_new_from_mem_buffer (GBytes *mem_buffer)
{
  GGMLContext *context = g_new0 (GGMLContext, 1);
  context->mem_buffer = g_bytes_ref (mem_buffer);

  size_t mem_buffer_size;
  gpointer mem_buffer_ptr = (gpointer) g_bytes_get_data (context->mem_buffer, &mem_buffer_size);

  struct ggml_init_params params = {
    .mem_size = mem_buffer_size,
    .mem_buffer = mem_buffer_ptr,
    .no_alloc = FALSE,
  };

  context->ctx = ggml_init (params);
  context->ref_count = 1;

  g_assert (context->ctx != NULL);
  return context;
}

/**
 * ggml_context_new:
 * @memory_size: The size of the memory pool for this context
 *
 * Creates a new #GGMLContext and memory pool from which #GGMLTensor
 * objects can be allocated. The @memory_size parameter is important -
 * you need to properly estimate it upfront, for example by using
 * what you know about the model architecture to be allocated, or
 * how many elements you will have in the input etc.
 *
 * Returns: (transfer full): A new #GGMLContext
 */
GGMLContext *
ggml_context_new (size_t memory_size)
{
  g_autoptr(GBytes) mem_buffer = g_bytes_new_take (g_malloc (memory_size), memory_size);
  return ggml_context_new_from_mem_buffer (mem_buffer);
}

/**
 * ggml_context_ref: (skip)
 * @context: A #GGMLContext
 *
 * Increases the reference count on @context
 *
 * Returns: (transfer full): The @context
 */
GGMLContext *
ggml_context_ref (GGMLContext *context)
{
  ++context->ref_count;
  return context;
}

/**
 * ggml_context_unref: (skip)
 * @context: A #GGMLContext
 *
 * Decreases the reference count on @context. If the reference
 * count goes to zero, then the context and its memory pool will
 * be released.
 */
void
ggml_context_unref (GGMLContext *context)
{
  if (--context->ref_count == 0)
    {
      g_clear_pointer (&context->ctx, ggml_free);
      g_clear_pointer (&context->mem_buffer, g_bytes_unref);
      g_clear_pointer (&context, g_free);
    }
}

/**
 * ggml_context_new_tensor:
 * @context: A #GGMLContext
 * @data_type: A #GGMLDataType for the new tensor
 * @shape: (array length=n_dims): Shape of the tensor
 * @n_dims: Number of dimensions in the tensor shape
 *
 * Creates a new #GGMLTensor from the memory pool of @context
 * with shape @shape
 *
 * Returns: (transfer full): The #GGMLTensor
 */
GGMLTensor *
ggml_context_new_tensor (GGMLContext  *context,
                         GGMLDataType  data_type,
                         int64_t      *shape,
                         size_t        n_dims)
{
  return ggml_tensor_new (context, data_type, shape, n_dims);
}

/**
 * ggml_context_new_tensor_1d:
 * @context: A #GGMLContext
 * @data_type: A #GGMLDataType for the new tensor
 * @size: Size of the tensor
 *
 * Creates a new #GGMLTensor from the memory pool of @context.
 *
 * Returns: (transfer full): The #GGMLTensor
 */
GGMLTensor *
ggml_context_new_tensor_1d (GGMLContext *context, GGMLDataType data_type, size_t size)
{
  return ggml_tensor_new_1d (context, data_type, size);
}

/**
 * ggml_context_new_tensor_2d:
 * @context: A #GGMLContext
 * @data_type: A #GGMLDataType for the new tensor
 * @width: Size of the tensor's first dimension
 * @height: Size of the tensor's second dimension
 *
 * Creates a new #GGMLTensor from the memory pool of @context. The size
 * of the tensor is width times height.
 *
 * Returns: (transfer full): The #GGMLTensor
 */
GGMLTensor *
ggml_context_new_tensor_2d (GGMLContext *context, GGMLDataType data_type, size_t width, size_t height)
{
  return ggml_tensor_new_2d (context, data_type, width, height);
}

/**
 * ggml_context_new_tensor_3d:
 * @context: A #GGMLContext
 * @data_type: A #GGMLDataType for the new tensor
 * @width: Size of the tensor's first dimension
 * @height: Size of the tensor's second dimension
 * @depth: Size of the tensor's third timension
 *
 * Creates a new #GGMLTensor from the memory pool of @context. The size
 * of the tensor is width times height times depth.
 *
 * Returns: (transfer full): The #GGMLTensor
 */
GGMLTensor *
ggml_context_new_tensor_3d (GGMLContext *context, GGMLDataType data_type, size_t width, size_t height, size_t depth)
{
  return ggml_tensor_new_3d (context, data_type, width, height, depth);
}

/**
 * ggml_context_new_scalar_f32:
 * @context: A #GGMLContext
 * @value: A #float with the value
 *
 * Creates a new #GGMLTensor of shape 1 with the scalar value given
 * in @value
 *
 * Returns: (transfer full): A new #GGMLTensor
 */
GGMLTensor *
ggml_context_new_scalar_f32 (GGMLContext *context,
                             float value)
{
  return ggml_tensor_new_scalar_f32 (context, value);
}

G_DEFINE_BOXED_TYPE (GGMLContext, ggml_context, ggml_context_ref, ggml_context_unref);
