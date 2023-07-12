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

static inline int32_t product_i32 (int32_t *array, size_t n)
{
  int32_t product = 1;
  for (size_t i = 0; i < n; ++i)
    {
      product *= array[i];
    }

  return product;
}

static gboolean
ggml_model_load_weights_from_istream (GInputStream *istream,
                                      GGMLModel *model,
                                      char ***out_loaded_keys,
                                      GCancellable *cancellable,
                                      GError **error)
{
  g_autoptr(GPtrArray) loaded_keys = g_ptr_array_new_full (0, g_free);

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

      size_t bytes_per_element = ggml_size_of_data_type (ttype);
      size_t allocated_bytes = 0;
      char *tensor_data_ptr = ggml_tensor_get_data (tensor, &allocated_bytes);

      size_t expected_bytes = (tensor_definition_n_elements * bytes_per_element / ggml_tensor_block_size (tensor));
      if (expected_bytes != allocated_bytes)
        {
          g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED, "Tensor %s has allocation of %zu bytes, expected %zu bytes", name_buffer, allocated_bytes, expected_bytes);
          return FALSE;
        }

      /* Now we can read the tensor */
      if (!ggml_input_stream_read_exactly (istream, tensor_data_ptr, allocated_bytes, cancellable, error))
        {
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
ggml_estimate_transformer_model_memory (int32_t n_vocab, int32_t n_embd, int32_t n_head, int32_t n_layer, int32_t n_ctx, int32_t ftype)
{
  const int32_t head_dim = n_embd / n_head;
  const int32_t kv_heads = n_head;
  const int32_t kv_dim = kv_heads * head_dim;
  enum ggml_type wtype = ggml_ftype_to_ggml_type((enum ggml_ftype) (ftype));
  size_t ctx_size = 0;

  ctx_size += n_embd * ggml_type_sizef(GGML_TYPE_F32); // ln_f_g
  ctx_size += n_embd * ggml_type_sizef(GGML_TYPE_F32); // ln_f_b

  ctx_size += n_vocab * n_embd * ggml_type_sizef(wtype);         // wte
  ctx_size +=   n_ctx * n_embd * ggml_type_sizef(GGML_TYPE_F32); // wpe
  ctx_size += n_vocab * n_embd * ggml_type_sizef(wtype);         // lm_head

  ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_1_g
  ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_1_b

  ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_2_g
  ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_2_b

  ctx_size += n_layer * ((n_embd + 2 * kv_dim) * n_embd * 3 * ggml_type_sizef(wtype));         // c_attn_attn_w // TODO:
  ctx_size += n_layer * (       (n_embd + 2 * kv_dim) * 3 * ggml_type_sizef(GGML_TYPE_F32)); // c_attn_attn_b

  ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype));           // c_attn_proj_w
  ctx_size += n_layer * (       n_embd * ggml_type_sizef(GGML_TYPE_F32));   // c_attn_proj_b

  ctx_size += n_layer * (4 * n_embd * n_embd * ggml_type_sizef(wtype));         // c_mlp_fc_w
  ctx_size += n_layer * (       4 * n_embd * ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_fc_b

  ctx_size += n_layer * (4 * n_embd *n_embd * ggml_type_sizef(wtype));         // c_mlp_proj_w
  ctx_size += n_layer * (         n_embd * ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_proj_b

  ctx_size += n_ctx*n_layer * n_embd * ggml_type_sizef(GGML_TYPE_F32); // memory_k
  ctx_size += n_ctx*n_layer * n_embd * ggml_type_sizef(GGML_TYPE_F32); // memory_v

  ctx_size += (6 + 12 * n_layer) * 512; // object overhead

  return ctx_size;
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
  const int32_t n_embd = ggml_hyperparameters_get_int32 (hyperparameters, "n_embd");
  const int32_t n_layer = ggml_hyperparameters_get_int32 (hyperparameters, "n_layer");
  const int32_t n_ctx = ggml_hyperparameters_get_int32 (hyperparameters, "n_ctx");
  const int32_t n_vocab = ggml_hyperparameters_get_int32 (hyperparameters, "n_vocab");
  const int32_t n_head = ggml_hyperparameters_get_int32 (hyperparameters, "n_head");
  const int32_t ftype = ggml_hyperparameters_get_int32 (hyperparameters, "ftype");

  size_t memory_size = ggml_estimate_transformer_model_memory (n_vocab, n_embd, n_head, n_layer, n_ctx, ftype);
  g_autoptr (GGMLContext) context = ggml_context_new (memory_size);
  g_autoptr (GHashTable) flattened_desc = ggml_model_desc_node_flatten (model_desc_node);
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
  ggml_compute_graph_compute (compute_graph, output->owning_context, 2);

  return g_steal_pointer (&output);
}

G_DEFINE_BOXED_TYPE (GGMLModel, ggml_model, ggml_model_ref, ggml_model_unref)
