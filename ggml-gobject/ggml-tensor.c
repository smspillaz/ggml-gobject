/*
 * ggml-gobject/ggml-tensor.c
 *
 * Library code for ggml-tensor
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

#include <ggml-gobject/ggml-tensor.h>
#include <ggml-gobject/internal/ggml-context-internal.h>
#include <ggml-gobject/internal/ggml-tensor-internal.h>

GGMLTensor *
ggml_tensor_from_tensor (GGMLContext *context, struct ggml_tensor *base_tensor)
{
  GGMLTensor *tensor = g_new0 (GGMLTensor, 1);
  tensor->ref_count = 1;
  tensor->owning_context = ggml_context_ref (context);
  tensor->tensor = base_tensor;

  return tensor;
}

GGMLTensor *
ggml_tensor_new_1d (GGMLContext *context, GGMLDataType data_type, size_t size)
{
  return ggml_tensor_from_tensor (context,
                                  ggml_new_tensor_1d (context->ctx,
                                                      (enum ggml_type) data_type,
                                                      size));
}

GGMLTensor *
ggml_tensor_new_2d (GGMLContext *context, GGMLDataType data_type, size_t width, size_t height)
{
  return ggml_tensor_from_tensor (context,
                                  ggml_new_tensor_2d (context->ctx,
                                                      (enum ggml_type) data_type,
                                                      width,
                                                      height));
}

GGMLTensor *
ggml_tensor_new_3d (GGMLContext *context, GGMLDataType data_type, size_t width, size_t height, size_t depth)
{
  return ggml_tensor_from_tensor (context,
                                  ggml_new_tensor_3d (context->ctx,
                                                      (enum ggml_type) data_type,
                                                      width,
                                                      height,
                                                      depth));
}

GGMLTensor *
ggml_tensor_new_scalar_f32 (GGMLContext *context,
                            float value)
{
  return ggml_tensor_from_tensor (context, ggml_new_f32 (context->ctx, value));
}

/**
 * ggml_tensor_ref: (skip)
 * @tensor: A #GGMLTensor
 *
 * Increases the reference count on @tensor
 *
 * Returns: (transfer full): A #GGMLTensor
 */
GGMLTensor *
ggml_tensor_ref (GGMLTensor *tensor)
{
  ++tensor->ref_count;
  return tensor;
}

/**
 * ggml_tensor_unref: (skip)
 * @tensor: A #GGMLTensor
 *
 * Decreases the reference count on @tensor and cleans up
 * if the reference count goes to zero. Note that that underlying
 * tensor is allocated as part of the memory pool in its
 * #GGMLContext, so the memory will be released as part of the memory
 * pool when the context is freed. This function decreases the reference
 * count on the #GGMLContext
 */
void
ggml_tensor_unref (GGMLTensor *tensor)
{
  if (--tensor->ref_count == 0)
    {
      g_clear_pointer (&tensor->owning_context, ggml_context_unref);

      /* Tensor is part of the owning context's memory pool, so its
       * memory gets freed when the context goes away */
      tensor->tensor = NULL;

      g_clear_pointer (&tensor, g_free);
    }
}

/**
 * ggml_tensor_element_size:
 * @tensor: A #GGMLTensor
 *
 * Returns: The number of bytes per element of the @tensor
 */
size_t
ggml_tensor_element_size (GGMLTensor *tensor)
{
  return ggml_element_size (tensor->tensor);
}

/**
 * ggml_tensor_n_elements:
 * @tensor: A #GGMLTensor
 *
 * Returns: The number of elements in the @tensor
 */
size_t
ggml_tensor_n_elements (GGMLTensor *tensor)
{
  return ggml_nelements (tensor->tensor);
}

/**
 * ggml_tensor_block_size:
 * @tensor: A #GGMLTensor
 *
 * Returns: The block size of the tensor
 */
size_t
ggml_tensor_block_size (GGMLTensor *tensor)
{
  return ggml_blck_size (tensor->tensor->type);
}

/**
 * ggml_tensor_n_bytes:
 * @tensor: A #GGMLTensor
 *
 * Returns: The number of bytes the @tensor consumes
 */
size_t
ggml_tensor_n_bytes (GGMLTensor *tensor)
{
  return ggml_nbytes (tensor->tensor);
}

/**
 * ggml_tensor_set_data: (skip)
 * @tensor: A #GGMLTensor
 * @data: A pointer to some data
 * @size: The number of bytes to read
 *
 * Sets the data of the tensor. It is the caller's responsibility to
 * pass a buffer of the correct size.
 */
void
ggml_tensor_set_data (GGMLTensor *tensor, char *data, size_t size)
{
  memcpy(tensor->tensor->data, (const void *) data, size);
}

/**
 * ggml_tensor_get_data: (skip)
 * @tensor: A #GGMLTensor
 * @out_n_bytes: An out-parameter for the number of bytes in @tensor
 *
 * Get a view into the raw bytes of @tensor and writes the number of bytes
 * into @out_n_bytes . The data is not copied.
 *
 * Returns: (transfer none): A #char array with the bytes for this tensor data.
 */
char *
ggml_tensor_get_data (GGMLTensor *tensor,
                      size_t *out_n_bytes)
{
  gpointer data = ggml_get_data (tensor->tensor);
  *out_n_bytes =  ggml_tensor_n_bytes (tensor);

  return data;
}

/**
 * ggml_tensor_get_bytes:
 * @tensor: A #GGMLTensor
 *
 * Makes a copy of the tensor data and returns it as a #GBytes
 * which may be more suitable for bindings (but is quite inefficient
 * unless you wanted to copy the data).
 *
 * Returns: (transfer full): A #GBytes containing the tensor data
 */
GBytes *
ggml_tensor_get_bytes (GGMLTensor *tensor)
{
  size_t n_bytes = 0;
  const char *data = ggml_tensor_get_data (tensor, &n_bytes);

  return g_bytes_new (data, n_bytes);
}

/**
 * ggml_tensor_set_data_from_bytes:
 * @tensor: A #GGMLTensor
 * @bytes: (transfer none): A #GBytes with some data
 *
 * Sets the data of the tensor from @bytes.
 * It is the caller's responsibility to pass a @bytes of the correct size.
 */
void
ggml_tensor_set_data_from_bytes (GGMLTensor *tensor, GBytes *bytes)
{
  size_t size = 0;
  gconstpointer data = g_bytes_get_data (bytes, &size);

  ggml_tensor_set_data (tensor, (char *) data, size);
}

/**
 * ggml_tensor_set_data_from_int32_array:
 * @tensor: A #GGMLTensor
 * @array: (array length=n_elements): An array of #int32_t elements.
 * @n_elements: Number of elements in @array
 *
 * Set the data of @tensor from the int32 array in @array.
 *
 * It is an error to call this function on a tensor that is not
 * of type %GGML_DATA_TYPE_I32.
 */
void
ggml_tensor_set_data_from_int32_array (GGMLTensor *tensor,
                                       int32_t    *array,
                                       size_t      n_elements)
{
  g_assert (tensor->tensor->type == (enum ggml_type) GGML_DATA_TYPE_I32);

  ggml_tensor_set_data (tensor, (char *) array, n_elements * sizeof (int32_t));
}

/**
 * ggml_tensor_set_name:
 * @tensor: (transfer none): A #GGMLTensor
 * @name: A string with the tensor name
 *
 * Sets the tensor name from @name. The name length limit is 32 characters,
 * longer names will be truncated.
 */
void
ggml_tensor_set_name (GGMLTensor *tensor,
                      const char *name)
{
  ggml_set_name (tensor->tensor, name);
}

/**
 * ggml_tensor_get_name:
 * @tensor: (transfer none): A #GGMLTensor
 *
 * Returns: (transfer none): The tensor name
 */
const char *
ggml_tensor_get_name (GGMLTensor *tensor)
{
  return ggml_get_name (tensor->tensor);
}

/**
 * ggml_tensor_get_data_type:
 * @tensor: A #GGMLTensor
 *
 * Returns: A #GGMLDataType which is the data type of this tensor
 */
GGMLDataType
ggml_tensor_get_data_type (GGMLTensor *tensor)
{
  return (GGMLDataType) tensor->tensor->type;
}

/**
 * ggml_tensor_get_shape:
 * @tensor: A #GGMLTensor
 * @out_n_dims: (out): Number of dimensions
 *
 * Returns: (transfer none): An #int64_t array with the shape of the tensor. Note that in
 *          GGML, shapes go front-to-back, so the first element is typically the vector dimension,
 *          then the number of sequence items, then the batch size, etc.
 */
int64_t *
ggml_tensor_get_shape (GGMLTensor *tensor, size_t *out_n_dims)
{
  if (out_n_dims != NULL)
    {
      *out_n_dims = tensor->tensor->n_dims;
    }

  return tensor->tensor->ne;
}

/**
 * ggml_tensor_get_cgraph_children:
 * @tensor: (transfer none): A #GGMLTensor
 *
 * Get the child tensors of this @tensor according to the most recent computation graph.
 *
 * A child tensor of a tensor is one that is antecedent to it, eg, if a = b + c, then the children
 * of 'a' are 'b' and 'c'.
 *
 * Returns: (transfer full) (element-type GGMLTensor): A #GPtrArray of #GGMLTensor objects
 *          wrapping each of the antecedent children of this tensor according to the most recent
 *          compute graph.
 */
GPtrArray *
ggml_tensor_get_cgraph_children (GGMLTensor *tensor)
{
  g_autoptr(GPtrArray) children = g_ptr_array_new_full (2, (GDestroyNotify) ggml_tensor_unref);

  for (size_t i = 0; i < GGML_MAX_SRC; ++i)
    {
      if (tensor->tensor->src[i] != NULL)
        {
          /* XXX: This isn't strictly speaking correct -
          * tensor->owning_context->ctx might be different
          * from tensor->src's context, meaning that if
          * the context is unref'd then tensor->src's memory
          * goes away. */
          g_ptr_array_add (children,
                           ggml_tensor_from_tensor (tensor->owning_context, tensor->tensor->src[i]));
        }
    }

  return g_steal_pointer (&children);
}

/**
 * ggml_tensor_get_cgraph_perf_us:
 * @tensor: A #GGMLTensor
 *
 * Returns: The average number of microseconds spent on computation in the most recent
 *          compute graph.
 */
int64_t
ggml_tensor_get_cgraph_perf_us (GGMLTensor *tensor)
{
  return (int32_t) (tensor->tensor->perf_time_us / ((float) tensor->tensor->perf_runs));
}

G_DEFINE_BOXED_TYPE (GGMLTensor, ggml_tensor, ggml_tensor_ref, ggml_tensor_unref);
