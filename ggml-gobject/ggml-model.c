/*
 * ggml-gobject/ggml-model.c
 *
 * Library code for ggml-model
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

#include <ggml-gobject/ggml-model.h>
#include <ggml-gobject/internal/ggml-stream-internal.h>
#include <ggml-gobject/internal/ggml-tensor-internal.h>
#include <ggml-gobject/ggml-enum-types.h>

struct _GGMLModel {
  GGMLContext *owning_context;
  GHashTable *weights;
  GGMLModelForwardFunc forward_func;
  gpointer forward_func_user_data;
  GDestroyNotify forward_func_user_data_destroy;
  size_t ref_count;
};

/**
 * ggml_context_new_model_from_flattened_desc:
 * @context: A #GGMLContext
 * @flattened_desc: (element-type utf8 GGMLModelDescLeaf): A #GHashTable containing
 *                  key-value pairs of weight names and their descriptions.
 * @forward_func: (scope notified) (nullable): A #GGMLModelFowardFunc
 * @forward_func_user_data: (closure forward_func) (transfer full): The user data for @forward_func
 * @forward_func_user_data_destroy: (destroy forward_func): A #GDestroyNotify for forward_func
 *
 * Returns: (transfer full): A new #GGMLModel
 */
GGMLModel *
ggml_model_new_from_flattened_desc (GGMLContext *context,
                                    GHashTable  *flattened_desc,
                                    GGMLModelForwardFunc forward_func,
                                    gpointer forward_func_user_data,
                                    GDestroyNotify forward_func_user_data_destroy)
{
  GGMLModel *model = g_new0 (GGMLModel, 1);
  model->owning_context = ggml_context_ref (context);
  model->weights = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, (GDestroyNotify) ggml_tensor_unref);
  model->forward_func = forward_func;
  model->forward_func_user_data = forward_func_user_data;
  model->forward_func_user_data_destroy = forward_func_user_data_destroy;
  model->ref_count = 1;

  GHashTableIter iter;
  gpointer key, value;

  g_hash_table_iter_init (&iter, flattened_desc);
  while (g_hash_table_iter_next (&iter, &key, &value))
    {
      GGMLModelDescLeaf *leaf = value;
      GGMLTensor *tensor = NULL;

      if (leaf->n_dim == 1)
        {
          tensor = ggml_context_new_tensor_1d (context, leaf->type, leaf->dimensions[0]);
        }
      else if (leaf->n_dim == 2)
        {
          tensor = ggml_context_new_tensor_2d (context, leaf->type, leaf->dimensions[0], leaf->dimensions[1]);
        }
      else if (leaf->n_dim == 3)
        {
          tensor = ggml_context_new_tensor_3d (context, leaf->type, leaf->dimensions[0], leaf->dimensions[1], leaf->dimensions[2]);
        }

      g_assert (tensor != NULL);
      ggml_tensor_set_name (tensor, key);
      g_hash_table_insert (model->weights, g_strdup (key), tensor);
    }

  return model;
}

static inline int32_t
product_i32 (int32_t *array, size_t n)
{
  int32_t product = 1;
  for (size_t i = 0; i < n; ++i)
    {
      product *= array[i];
    }

  return product;
}

static inline int64_t
product_i64 (int64_t *array, size_t n)
{
  int64_t product = 1;
  for (size_t i = 0; i < n; ++i)
    {
      product *= array[i];
    }

  return product;
}

static GArray *
data_to_f32_array (GGMLDataType   data_type,
                   char          *data,
                   size_t         n_elements,
                   float        **out_data_ptr)
{
  g_assert (data_type == GGML_DATA_TYPE_F16 || data_type == GGML_DATA_TYPE_F32);

  if (data_type == GGML_DATA_TYPE_F16)
    {
      g_autoptr(GArray) tensor_data = g_array_sized_new (FALSE, FALSE, sizeof (float), n_elements);
      tensor_data->len = n_elements;

      ggml_fp16_t *fp16_data = (ggml_fp16_t *) data;

      for (size_t i = 0; i < n_elements; ++i)
        {
          g_array_index (tensor_data, float, i) = ggml_fp16_to_fp32 (fp16_data[i]);
        }

      *out_data_ptr = (float *) tensor_data->data;
      return g_steal_pointer (&tensor_data);
    }

  /* In this case, we just return NULL and instead
   * alias the data directly */
  *out_data_ptr = (float *) data;

  return NULL;
}

static size_t
convert_f32_to_f16 (const float   *data,
                    size_t         n_elements,
                    char          *out_data_ptr)
{
  ggml_fp16_t *fp16_ptr = (ggml_fp16_t *) out_data_ptr;
  const float *fp32_ptr = (const float *) data;

  for (size_t i = 0; i < n_elements; ++i)
    {
      fp16_ptr[i] = fp32_ptr[i];
    }

  return n_elements * ggml_type_size ((enum ggml_type) GGML_DATA_TYPE_F16);
}

/**
 * ggml_convert_data: (skip)
 * @src_type: The source #GGMLDataType
 * @original_data: The original data in bytes
 * @original_data_length: The length of the original data
 * @shape: The shape of the original data
 * @n_dims: The number of dimensions in the original data's shape
 * @quantize_type: The target #GGMLDataType to quantize into
 * @histogram: (inout) (array length=n_histogram): A histogram to write into
 * @n_histogram: Length of @histogram
 * @out_data: (inout): An output pointer for quantized data
 * @out_data_len: The size of the output quantized data.
 * @error: A #GError
 *
 * Quantize tensor data or convert to another data type.
 * Histogram information is written into @histogram if it is set.
 *
 * Returns: %TRUE on success, %FALSE with @error set on failure.
 */
static gboolean
convert_data_for_model (GGMLDataType   src_type,
                        char          *original_data,
                        size_t         original_data_length,
                        int64_t       *shape,
                        size_t         n_dims,
                        GGMLDataType   tgt_type,
                        int64_t       *histogram,
                        size_t         n_histogram,
                        char          *out_data,
                        size_t         out_data_len,
                        GError       **error)
{
  /* This should not usually happen, but in this case we memcpy
   * directly into the out_data ptr, assuming that it is not aliased */
  if (src_type == tgt_type)
    {
      if (out_data == original_data)
        {
          return TRUE;
        }

      if (out_data_len != original_data_length)
        {
          g_set_error (error,
                       G_IO_ERROR,
                       G_IO_ERROR_FAILED,
                       "Cannot copy from src to tgt, buffer sizes (src: %zu, tgt: %zu) differ",
                       original_data_length,
                       out_data_len);
          return FALSE;
        }

      memcpy (out_data, original_data, out_data_len);
      return TRUE;
    }

  if (src_type != GGML_DATA_TYPE_F32 &&
      src_type != GGML_DATA_TYPE_F16)
    {
      g_autoptr(GEnumClass) src_type_enum = g_type_class_ref (GGML_TYPE_DATA_TYPE);
      GEnumValue *src_data_type_value = g_enum_get_value (src_type_enum, src_type);
      g_set_error (error,
                   G_IO_ERROR,
                   G_IO_ERROR_FAILED,
                   "Cannot convert from src_type %s, src_type must be F32 or F16",
                   src_data_type_value->value_name);
      return FALSE;
    }

  float *data_f32_ptr = NULL;

  /* We return data_f32 here, but it might not be defined - its
   * just a convenience to free the data later, but the real
   * data pointer comes from data_ptr. This helps to avoid copies where
   * not necessary. */
  size_t n_elements = product_i64 (shape, n_dims);
  g_autoptr(GArray) data_f32 = data_to_f32_array (src_type,
                                                  original_data,
                                                  n_elements,
                                                  &data_f32_ptr);

  size_t cur_size = 0;

  switch (tgt_type)
    {
      case GGML_DATA_TYPE_Q4_0:
        cur_size = ggml_quantize_q4_0 ((float *) data_f32_ptr,
                                       out_data,
                                       n_elements,
                                       shape[0],
                                       histogram);
        break;
      case GGML_DATA_TYPE_Q4_1:
        cur_size = ggml_quantize_q4_1 ((float *) data_f32_ptr,
                                       out_data,
                                       n_elements,
                                       shape[0],
                                       histogram);
        break;
      case GGML_DATA_TYPE_Q5_0:
        cur_size = ggml_quantize_q5_0 ((float *) data_f32_ptr,
                                       out_data,
                                       n_elements,
                                       shape[0],
                                       histogram);
        break;
      case GGML_DATA_TYPE_Q5_1:
        cur_size = ggml_quantize_q5_1 ((float *) data_f32_ptr,
                                       out_data,
                                       n_elements,
                                       shape[0],
                                       histogram);
        break;
      case GGML_DATA_TYPE_Q8_0:
        cur_size = ggml_quantize_q8_0 ((float *) data_f32_ptr,
                                       out_data,
                                       n_elements,
                                       shape[0],
                                       histogram);
        break;
      case GGML_DATA_TYPE_F16:
        cur_size = convert_f32_to_f16 ((float *) data_f32_ptr,
                                       n_elements,
                                       out_data);
        break;
      default:
        {
          g_autoptr(GEnumClass) tgt_type_class = g_type_class_ref (GGML_TYPE_DATA_TYPE);
          GEnumValue *tgt_data_type_value = g_enum_get_value (tgt_type_class, tgt_type);
          g_set_error (error,
                       G_IO_ERROR,
                       G_IO_ERROR_FAILED,
                       "Conversion failed, tgt_type cannot be %s",
                       tgt_data_type_value->value_name);
          return FALSE;
        }
    }

  g_assert (cur_size <= out_data_len);

  return TRUE;
}

static gboolean
read_into_tensor (GGMLTensor    *tensor,
                  GGMLDataType   stream_data_type,
                  GInputStream  *istream,
                  int64_t       *histogram,
                  size_t         histogram_len,
                  GCancellable  *cancellable,
                  GError       **error)
{
  GGMLDataType tensor_data_type = ggml_tensor_get_data_type (tensor);
  size_t tensor_definition_n_elements = ggml_tensor_n_elements (tensor);
  size_t stream_bytes_per_element = ggml_size_of_data_type (stream_data_type);
  size_t expected_bytes = (tensor_definition_n_elements * stream_bytes_per_element / ggml_blck_size ((enum ggml_type) stream_data_type));
  size_t allocated_bytes = 0;
  char *tensor_data_ptr = ggml_tensor_get_data (tensor, &allocated_bytes);

  if (stream_data_type != tensor_data_type)
    {
      /* Conversion required. First read the data from the stream,
       * then convert it and write the result into the tensor */
      g_autoptr(GArray) stream_data = g_array_sized_new (FALSE, TRUE, sizeof (char), expected_bytes);
      stream_data->len = expected_bytes;

      /* Now we can read the tensor data */
      if (!ggml_input_stream_read_exactly (istream,
                                           stream_data->data,
                                           expected_bytes,
                                           cancellable,
                                           error))
        {
          return FALSE;
        }

      /* Now apply the conversion required */
      GError *my_error = NULL;
      size_t n_dims;
      int64_t *shape = ggml_tensor_get_shape (tensor, &n_dims);

      if (!convert_data_for_model (stream_data_type,
                                   stream_data->data,
                                   stream_data->len,
                                   shape,
                                   n_dims,
                                   tensor_data_type,
                                   histogram,
                                   histogram_len,
                                   tensor_data_ptr,
                                   allocated_bytes,
                                   &my_error))
        {
          g_set_error (error,
                       G_IO_ERROR,
                       G_IO_ERROR_FAILED,
                       "Unable to convert %s",
                       my_error->message);
          g_clear_error (&my_error);
          return FALSE;
        }

      return TRUE;
    }

  if (expected_bytes != allocated_bytes)
    {
      g_set_error (error,
                   G_IO_ERROR,
                   G_IO_ERROR_FAILED,
                   "Tensor allocation of %zu bytes, expected %zu bytes",
                   allocated_bytes,
                   expected_bytes);
      return FALSE;
    }

  /* No conversion required, just read the tensor */
  if (!ggml_input_stream_read_exactly (istream,
                                       tensor_data_ptr,
                                       allocated_bytes,
                                       cancellable,
                                       error))
    {
      return FALSE;
    }

  return TRUE;
}

static gboolean
ggml_model_load_weights_from_istream (GInputStream *istream,
                                      GGMLModel *model,
                                      char ***out_loaded_keys,
                                      GCancellable *cancellable,
                                      GError **error)
{
  g_autoptr(GPtrArray) loaded_keys = g_ptr_array_new_full (0, g_free);
  g_autoptr(GArray) histogram = g_array_sized_new (FALSE, TRUE, sizeof (int64_t), 1 << 4);
  histogram->len = 1 << 4;

  while (TRUE)
    {
      size_t bytes_read = 0;
      int32_t n_dims = 0;
      int32_t name_length = 0;
      int32_t ttype = 0;

      g_assert (n_dims <= 2);

      if (!g_input_stream_read_all (istream, (char *) &n_dims, sizeof (int32_t) * 1, &bytes_read, cancellable, error))
        {
          return FALSE;
        }

      /* As an exemption, if we don't read anything here, we're at the end of the stream, eg, we're done
       * and break out of this loop */
      if (bytes_read == 0)
        {
          break;
        }

      if (!ggml_input_stream_read_exactly (istream, (char *) &name_length, sizeof (int32_t) * 1, cancellable, error))
        {
          return FALSE;
        }

      if (!ggml_input_stream_read_exactly (istream, (char *) &ttype, sizeof (int32_t) * 1, cancellable, error))
        {
          return FALSE;
        }

      int dims_buffer[2];

      if (!ggml_input_stream_read_exactly (istream, (char *) dims_buffer, sizeof (int32_t) * n_dims, cancellable, error))
        {
          return FALSE;
        }

      int32_t input_stream_tensor_n_elements = product_i32(dims_buffer, n_dims);
      g_autofree char *name_buffer = g_new0 (char, name_length + 1);

      if (!ggml_input_stream_read_exactly (istream, (char *) name_buffer, sizeof (char) * name_length, cancellable, error))
        {
          return FALSE;
        }

      name_buffer[name_length] = '\0';

      /* Lookup tensor in the model weights. If its not there, then we have an error. */
      GGMLTensor *tensor = ggml_model_get (model, name_buffer);

      if (tensor == NULL)
        {
          g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED, "Tensor %s not found in model definition", name_buffer);
          return FALSE;
        }

      /* We did find the tensor, lets check that the size matches */
      size_t tensor_definition_n_elements = ggml_tensor_n_elements (tensor);

      if (tensor_definition_n_elements != input_stream_tensor_n_elements)
        {
          g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED, "Tensor %s had %zu elements in its definition, but the input stream has %d elements", name_buffer, tensor_definition_n_elements, input_stream_tensor_n_elements);
          return FALSE;
        }

      GError *my_error = NULL;

      if (!read_into_tensor (tensor,
                             ttype,
                             istream,
                             (int64_t *) histogram->data,
                             histogram->len,
                             cancellable,
                             &my_error))
        {
          g_set_error (error,
                       G_IO_ERROR,
                       G_IO_ERROR_FAILED,
                       "Unable to read into tensor %s: %s",
                       name_buffer,
                       my_error->message);
          g_clear_error (&my_error);
          return FALSE;
        }

      g_ptr_array_add (loaded_keys, g_strdup (name_buffer));
    }

  /* Add sentinel */
  g_ptr_array_add (loaded_keys, NULL);

  if (out_loaded_keys != NULL)
    {
      *out_loaded_keys = (char **) g_ptr_array_steal (loaded_keys, NULL);
    }

  return TRUE;
}

static size_t
ggml_estimate_tensor_size_for_type (GGMLDataType  data_type,
                                    int64_t      *shape,
                                    size_t        n_shape)
{
  size_t nb[GGML_MAX_DIMS];
  size_t ne[GGML_MAX_DIMS];
  size_t blk_size = ggml_blck_size (data_type);

  /* The true size is the stride at the final shape index * number
   * of elements (eg, how do we get from the beginning to the end
   * of the tensor on the outer dimension */
  nb[0] = ggml_type_size (data_type);
  ne[0] = shape[0];
  nb[1] = nb[0] * (ne[0] / blk_size);
  ne[1] = (n_shape > 1) ? shape[1] : 1;

  for (size_t i = 2; i < GGML_MAX_DIMS; ++i)
    {
      ne[i] = (i < n_shape) ? shape[i] : 1;
      nb[i] = ne[i - 1] * nb[i - 1];
    }

  return ne[GGML_MAX_DIMS - 1] * nb[GGML_MAX_DIMS - 1];
}

static size_t
ggml_estimate_model_size_from_flattened_desc (GHashTable *flattened_desc)
{
  gpointer key, value;
  GHashTableIter iter;

  size_t computed_size = 0;
  size_t overhead = ggml_tensor_overhead ();

  g_hash_table_iter_init (&iter, flattened_desc);
  while (g_hash_table_iter_next (&iter, &key, &value))
    {
      GGMLModelDescLeaf *leaf = value;

      size_t n_dims = leaf->n_dim;
      int64_t *shape = leaf->dimensions;

      /* Here we will quantize, so estimate the size of the quantized tensor */
      computed_size += ggml_estimate_tensor_size_for_type (leaf->type,
                                                           shape,
                                                           n_dims) + overhead;
    }

  return computed_size;
}

/**
 * ggml_model_load_from_istream:
 * @istream: (transfer none): A #GInputStream
 * @model_desc_node: (transfer none): A #GGMLModelDescNode
 * @hyperparameters: (transfer none): A #GGMLHyperparameters
 * @forward_func: A #GGMLModelForwardFunc
 * @forward_func_user_data: (closure forward_func): A user-data closure for @forward_func
 * @forward_func_user_data_destroy: (destroy forward_func): A #GDestroyNotify for @forward_func_user_data
 * @out_loaded_keys: (out) (transfer full) (nullable): A #GStrv out-parameter for the loaded keys.
 * @cancellable: (transfer none) (nullable): A #GCancellable
 * @error: A #GError out-parameter
 *
 * Returns: (transfer full): A new #GGMLModel with structure @model_desc_node,
 *                           loaded from @istream or %NULL with @error set on failure.
 */
GGMLModel *
ggml_model_load_from_istream (GInputStream                           *istream,
                              GGMLModelDescNode                      *model_desc_node,
                              GGMLHyperparameters                    *hyperparameters,
                              GGMLModelForwardFunc                    forward_func,
                              gpointer                                forward_func_user_data,
                              GDestroyNotify                          forward_func_user_data_destroy,
                              char                                 ***out_loaded_keys,
                              GCancellable                           *cancellable,
                              GError                                **error)
{
  g_autoptr (GHashTable) flattened_desc = ggml_model_desc_node_flatten (model_desc_node);
  size_t memory_size = ggml_estimate_model_size_from_flattened_desc (flattened_desc);
  g_autoptr (GGMLContext) context = ggml_context_new (memory_size);
  g_autoptr (GGMLModel) model = ggml_model_new_from_flattened_desc (context,
                                                                    flattened_desc,
                                                                    forward_func,
                                                                    forward_func_user_data,
                                                                    forward_func_user_data_destroy);

  /* Now that we have the model, we can start loading in the weights */
  if (!ggml_model_load_weights_from_istream (istream, model, out_loaded_keys, cancellable, error))
    {
      return FALSE;
    }

  return g_steal_pointer (&model);
}

typedef struct _GGMLModelLoadFromIstreamData
{
  GInputStream *istream;
  GGMLHyperparameters *hyperparameters;
  GGMLModelDescNode *model_desc_node;
  GGMLModelForwardFunc forward_func;
  gpointer forward_func_user_data;
  GDestroyNotify forward_func_user_data_destroy;
} GGMLModelLoadFromIstreamData;

static GGMLModelLoadFromIstreamData *
ggml_model_load_from_istream_data_new (GInputStream *istream,
                                       GGMLModelDescNode *model_desc_node,
                                       GGMLHyperparameters *hyperparameters,
                                       GGMLModelForwardFunc forward_func,
                                       gpointer forward_func_user_data,
                                       GDestroyNotify forward_func_user_data_destroy)
{
  GGMLModelLoadFromIstreamData *data = g_new0 (GGMLModelLoadFromIstreamData, 1);
  data->istream = g_object_ref (istream);
  data->model_desc_node = ggml_model_desc_node_ref (model_desc_node);
  data->hyperparameters = ggml_hyperparameters_ref (hyperparameters);
  data->forward_func = forward_func;
  data->forward_func_user_data = forward_func_user_data;
  data->forward_func_user_data_destroy = forward_func_user_data_destroy;

  return data;
}

static void
ggml_model_load_from_istream_data_free (GGMLModelLoadFromIstreamData *data)
{
  g_clear_pointer (&data->istream, g_object_unref);
  g_clear_pointer (&data->model_desc_node, ggml_model_desc_node_unref);
  g_clear_pointer (&data->hyperparameters, ggml_hyperparameters_unref);
  g_clear_pointer (&data->forward_func_user_data, data->forward_func_user_data_destroy);
  g_clear_pointer (&data, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLModelLoadFromIstreamData, ggml_model_load_from_istream_data_free);

typedef struct _GGMLModelLoadFromIstreamResult
{
  GGMLModel *model;
  GStrv out_loaded_keys;
} GGMLModelLoadFromIstreamResult;

GGMLModelLoadFromIstreamResult *
ggml_model_load_from_istream_result_new (GGMLModel *model,
                                         GStrv out_loaded_keys)
{
  GGMLModelLoadFromIstreamResult *result = g_new0 (GGMLModelLoadFromIstreamResult, 1);
  result->model = ggml_model_ref (model);
  result->out_loaded_keys = out_loaded_keys; /* transfer full */

  return result;
}

void
ggml_model_load_from_istream_result_free (GGMLModelLoadFromIstreamResult *result)
{
  g_clear_pointer (&result->model, ggml_model_unref);
  g_clear_pointer (&result->out_loaded_keys, g_strfreev);
  g_clear_pointer (&result, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLModelLoadFromIstreamResult, ggml_model_load_from_istream_result_free);

static void
ggml_model_load_from_istream_async_thread (GTask         *task,
                                           gpointer       source_object,
                                           gpointer       task_data,
                                           GCancellable  *cancellable)
{
  GGMLModelLoadFromIstreamData *data = task_data;
  g_auto(GStrv) out_loaded_keys = NULL;
  GError *error = NULL;

  g_autoptr(GGMLModel) model = ggml_model_load_from_istream (data->istream,
                                                             data->model_desc_node,
                                                             data->hyperparameters,
                                                             data->forward_func,
                                                             data->forward_func_user_data,
                                                             data->forward_func_user_data_destroy,
                                                             &out_loaded_keys,
                                                             cancellable,
                                                             &error);

  if (model == NULL)
    {
      g_task_return_error (task, error);
      return;
    }

  /* Transfer the forward func to the model */
  data->forward_func = NULL;
  data->forward_func_user_data = NULL;
  data->forward_func_user_data_destroy = NULL;

  /* Need to create a container here */
  g_autoptr(GGMLModelLoadFromIstreamResult) result = ggml_model_load_from_istream_result_new (model,
                                                                                              g_steal_pointer (&out_loaded_keys));

  g_task_return_pointer (task, g_steal_pointer (&result), (GDestroyNotify) ggml_model_load_from_istream_result_free);
}

GGMLModel *
ggml_model_load_from_istream_finish (GAsyncResult  *result,
                                     char        ***out_loaded_keys,
                                     GError       **error)
{
  g_return_val_if_fail (g_task_is_valid (result, NULL), NULL);
  GTask *task = G_TASK (result);

  g_autoptr(GGMLModelLoadFromIstreamResult) task_result = g_task_propagate_pointer (task, error);

  if (task_result == NULL)
    {
      return NULL;
    }

  *out_loaded_keys = g_steal_pointer (&task_result->out_loaded_keys);
  return g_steal_pointer (&task_result->model);
}

void
ggml_model_load_from_istream_async (GInputStream *istream,
                                    GGMLModelDescNode *model_desc,
                                    GGMLHyperparameters *hyperparameters,
                                    GGMLModelForwardFunc forward_func,
                                    gpointer forward_func_user_data,
                                    GDestroyNotify forward_func_user_data_destroy,
                                    GCancellable *cancellable,
                                    GAsyncReadyCallback callback,
                                    gpointer user_data)
{
  g_autoptr(GGMLModelLoadFromIstreamData) data = ggml_model_load_from_istream_data_new(istream,
                                                                                       model_desc,
                                                                                       hyperparameters,
                                                                                       forward_func,
                                                                                       forward_func_user_data,
                                                                                       forward_func_user_data_destroy);

  g_autoptr(GTask) task = g_task_new (NULL, cancellable, callback, user_data);
  g_task_set_task_data (task, g_steal_pointer (&data), (GDestroyNotify) ggml_model_load_from_istream_data_free);
  g_task_run_in_thread (task, ggml_model_load_from_istream_async_thread);
}

/**
 * ggml_model_get:
 * @model: A #GGMLModel
 * @key: A key to look up in the model
 *
 * Returns: (transfer none): The #GGMLTensor corresponding to the @key in the
 *          model, or %NULL if it does not exist.
 */
GGMLTensor *
ggml_model_get (GGMLModel *model, const char *key)
{
  return g_hash_table_lookup (model->weights, key);
}

/**
 * ggml_model_ref:
 * @model: A #GGMLModel
 *
 * Increase the reference count on @model.
 *
 * Returns: (transfer full): A #GGMLModel
 */
GGMLModel *
ggml_model_ref (GGMLModel *model)
{
  ++model->ref_count;

  return model;
}

/**
 * ggml_model_unref:
 * @model: A #GGMLModel
 *
 * Decreases the ref count on @model. If the count drops to zero, then
 * the model will be freed. This will drop the reference on the model's
 * underlying context, but most of the memory will not be freed until the
 * context goes away and its memory pool is cleaned up.
 */
void
ggml_model_unref (GGMLModel *model)
{
  if (--model->ref_count == 0)
    {
      g_clear_pointer (&model->owning_context, ggml_context_unref);
      g_clear_pointer (&model->weights, g_hash_table_destroy);

      if (model->forward_func_user_data_destroy)
        {
          g_clear_pointer (&model->forward_func_user_data, model->forward_func_user_data_destroy);
        }

      g_clear_pointer (&model, g_free);
    }
}

/**
 * ggml_model_forward:
 * @model: (transfer none): A #GGMLModel
 * @hyperparameters: (transfer none) (nullable): A #GGMLHyperparameters for the model
 * @inputs: (transfer none): An #GVariant with some inputs
 * @forward_parameters: (element-type utf8 int) (transfer none) (nullable): A #GHashTable with evaluation-specific parameters
 * @mem_buffer: (transfer none) (nullable): A #GBytes memory buffer that can be re-used.
 * @cancellable: (transfer none) (nullable): A #GCancellable
 * @error: A #GError out-parameter
 *
 * Does a forward pass on the model to define the compute graph, then runs the computation.
 *
 * Returns: (transfer full): A #GGMLTensor that can be used to create a #GGMLComputeGraph.
*/
GGMLTensor *
ggml_model_forward (GGMLModel *model,
                    GGMLHyperparameters *hyperparameters,
                    GVariant *inputs,
                    GHashTable *forward_parameters,
                    GBytes   *mem_buffer,
                    GCancellable *cancellable,
                    GError **error)
{
  g_autoptr(GGMLComputeGraph) compute_graph = ggml_compute_graph_new ();
  g_autoptr(GGMLTensor) output = (*model->forward_func) (model,
                                                         hyperparameters,
                                                         inputs,
                                                         forward_parameters,
                                                         compute_graph,
                                                         mem_buffer,
                                                         model->forward_func_user_data,
                                                         error);

  if (output == NULL)
    {
      return NULL;
    }

  ggml_compute_graph_build_forward_expand (compute_graph, output);

  size_t num_threads = g_get_num_processors ();
  g_autoptr(GGMLComputePlan) compute_plan = ggml_compute_graph_plan (compute_graph, num_threads);

  if (!ggml_compute_graph_compute (compute_graph,
                                   compute_plan,
                                   output->owning_context,
                                   cancellable,
                                   error))
    {
      return NULL;
    }

  return g_steal_pointer (&output);
}

G_DEFINE_BOXED_TYPE (GGMLModel, ggml_model, ggml_model_ref, ggml_model_unref)
