/*
 * ggml-gobject/ggml-gobject.c
 *
 * Library code for ggml-gobject
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

#include <ggml-gobject/ggml-gobject.h>
#include <gio/gio.h>

typedef struct _GGMLModelDescLeafExtended  {
  GGMLModelDescLeaf base;
  size_t ref_count;
} GGMLModelDescLeafExtended;

/**
 * ggml_model_desc_leaf_new:
 * @dimensions: (array length=n_dim): An #int64_t array with leaf node dimensions
 * @n_dim: Number of dimensions in @dimensions
 * @type: A #GGMLDataType for this leaf node
 *
 * Returns: (transfer full): A new #GGMLModelDescLeaf
 */
GGMLModelDescLeaf *
ggml_model_desc_leaf_new (int64_t *dimensions, size_t n_dim, GGMLDataType type)
{
  GGMLModelDescLeafExtended *leaf = (GGMLModelDescLeafExtended *) g_new0 (GGMLModelDescLeafExtended, 1);
  leaf->base.dimensions = g_new0 (int64_t, n_dim);
  leaf->base.n_dim = n_dim;
  leaf->base.type = type;
  leaf->ref_count = 1;

  memcpy (leaf->base.dimensions, dimensions, sizeof(int64_t) * leaf->base.n_dim);

  return (GGMLModelDescLeaf *) leaf;
}

/**
 * ggml_model_desc_leaf_ref:
 * @leaf: (transfer none): A #GGMLModelDescLeaf
 *
 * Returns: (transfer full): A new #GGMLModelDescLeaf
 */
GGMLModelDescLeaf *
ggml_model_desc_leaf_ref (GGMLModelDescLeaf *src)
{
  GGMLModelDescLeafExtended *ext = (GGMLModelDescLeafExtended *) src;

  ++ext->ref_count;
  return src;
}

void
ggml_model_desc_leaf_unref (GGMLModelDescLeaf *leaf)
{
  GGMLModelDescLeafExtended *ext = (GGMLModelDescLeafExtended *) leaf;

  if (--ext->ref_count == 0)
    {
      g_clear_pointer (&ext->base.dimensions, g_free);
      g_clear_pointer (&ext, g_free);
    }
}

G_DEFINE_BOXED_TYPE (GGMLModelDescLeaf,
                     ggml_model_desc_leaf,
                     ggml_model_desc_leaf_ref,
                     ggml_model_desc_leaf_unref);

typedef struct _GGMLModelDescNodeExtended {
  GGMLModelDescNode base;
  size_t ref_count;
} GGMLModelDescNodeExtended;

static GHashTable *
copy_hash_table (GHashTable *src, GCopyFunc key_copy_func, gpointer key_copy_func_user_data, GCopyFunc value_copy_func, gpointer value_copy_func_user_data)
{
  GHashTable *dst = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, (GDestroyNotify) ggml_model_desc_node_unref);

  GHashTableIter iter;
  gpointer key, value;

  g_hash_table_iter_init (&iter, src);
  while (g_hash_table_iter_next (&iter, &key, &value))
    {
      g_hash_table_insert (dst,
                           key_copy_func (key, key_copy_func_user_data),
                           value_copy_func (value, value_copy_func_user_data));
    }

  return dst;
}

/**
 * ggml_model_desc_node_new:
 * @leaf: (transfer none) (nullable): The leaf to initialize this node with.
 * @children: (transfer none) (nullable) (element-type utf8 GGMLModelDescNode): The list of children in this node.
 *
 * Returns: (transfer full): A new #GGMLModelNode
 */
GGMLModelDescNode *
ggml_model_desc_node_new (GGMLModelDescLeaf *leaf, GHashTable *children)
{
  GGMLModelDescNodeExtended *node = (GGMLModelDescNodeExtended *) g_new0 (GGMLModelDescNodeExtended, 1);

  if (leaf != NULL)
    {
      node->base.leaf = ggml_model_desc_leaf_ref (leaf);
    }

  if (children != NULL)
    {
      node->base.children = copy_hash_table (children,
                                        (GCopyFunc) g_strdup,
                                        NULL,
                                        (GCopyFunc) ggml_model_desc_node_ref,
                                        NULL);
    }
  else
    {
      node->base.children = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, (GDestroyNotify) ggml_model_desc_node_unref);
    }

  node->ref_count = 1;
  return (GGMLModelDescNode *) node;
}

/**
 * ggml_model_desc_node_new_leaf:
 * @dimensions: (array length=n_dim): An #int64_t array with leaf node dimensions
 * @n_dim: Number of dimensions in @dimensions
 * @type: A #GGMLDataType for this leaf node
 *
 * Returns: (transfer full): A new #GGMLModelDescNode with a new #GGMLModelDescLeaf and no children
 */
GGMLModelDescNode *
ggml_model_desc_node_new_leaf (int64_t *dimensions, size_t n_dim, GGMLDataType type)
{
  g_autoptr(GGMLModelDescLeaf) leaf = ggml_model_desc_leaf_new (dimensions, n_dim, type);
  GGMLModelDescNode *node = ggml_model_desc_node_new (leaf, NULL);

  return node;
}

void
ggml_model_node_flatten_recurse (GHashTable *table, GGMLModelDescNode *current_node, const gchar *current_path)
{
  if (current_node->leaf != NULL)
    {
      g_hash_table_insert (table, g_strdup (current_path), ggml_model_desc_leaf_ref (current_node->leaf));
    }

  if (current_node->children != NULL)
    {
      GHashTableIter iter;
      gpointer key, value;

      g_hash_table_iter_init (&iter, current_node->children);

      while (g_hash_table_iter_next (&iter, &key, &value))
        {
          g_autofree gchar * next_path = NULL;
          next_path = (current_path == NULL ? g_strdup(key) : g_strjoin("/", current_path, (const gchar *) key, NULL));

          ggml_model_node_flatten_recurse (table, (GGMLModelDescNode *) value, next_path);
        }
    }
}

/**
 * ggml_model_desc_node_flatten:
 * @node: A #GGMLModelDescNode to flatten
 *
 * Returns: (transfer full) (element-type utf8 GGMLModelDescLeaf): A flattened tree with all
 * the nodes, each with slash-namespaced names.
 */
GHashTable *
ggml_model_desc_node_flatten (GGMLModelDescNode *node)
{
  GHashTable *ht = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, (GDestroyNotify) ggml_model_desc_leaf_unref);
  ggml_model_node_flatten_recurse (ht, node, NULL);

  return ht;
}

/**
 * ggml_model_desc_node_ref:
 * @node: (transfer none): A #GGMLModelDescNode
 *
 * Recursively copies the @node
 *
 * Returns: (transfer full): A new #GGMLModelDescNode
 */
GGMLModelDescNode *
ggml_model_desc_node_ref (GGMLModelDescNode *src)
{
  GGMLModelDescNodeExtended *ext = (GGMLModelDescNodeExtended *) src;
  ++ext->ref_count;

  return src;
}

/**
 * ggml_model_desc_node_unref:
 * @node: (transfer full): A #GGMLModelDescNode
 *
 * Recursively frees the @node
 */
void
ggml_model_desc_node_unref (GGMLModelDescNode *node)
{
  GGMLModelDescNodeExtended *ext = (GGMLModelDescNodeExtended *) node;

  if (--ext->ref_count == 0)
    {
      g_clear_pointer (&node->leaf, ggml_model_desc_leaf_unref);
      g_clear_pointer (&node->children, g_hash_table_destroy);
      g_clear_pointer (&node, g_free);
    }
}

G_DEFINE_BOXED_TYPE (GGMLModelDescNode,
                     ggml_model_desc_node,
                     ggml_model_desc_node_ref,
                     ggml_model_desc_node_unref)

struct _GGMLContext {
  struct ggml_context *ctx;
  size_t ref_count;
};

struct _GGMLTensor {
  GGMLContext *owning_context;
  struct ggml_tensor *tensor;
  size_t ref_count;
};

struct _GGMLModel {
  GGMLContext *owning_context;
  GHashTable *weights;
  GGMLModelForwardFunc forward_func;
  gpointer forward_func_user_data;
  GDestroyNotify forward_func_user_data_destroy;
  size_t ref_count;
};

struct _GGMLHyperparameters {
  gchar **ordered_keys;
  GHashTable *parameters;
  size_t ref_count;
};

struct _GGMLTokenDictionary {
  gchar **idx_to_word;
  GHashTable *word_to_idx;
  size_t ref_count;
};

struct _GGMLLanguageModel {
  GGMLHyperparameters *hyperparameters;
  GGMLTokenDictionary *token_dictionary;
  GGMLModel *model;
  size_t ref_count;
};

struct _GGMLComputeGraph {
  struct ggml_cgraph cgraph;
  size_t ref_count;
};

static GGMLTensor *
ggml_tensor_from_tensor (GGMLContext *context, struct ggml_tensor *base_tensor)
{
  GGMLTensor *tensor = g_new0 (GGMLTensor, 1);
  tensor->ref_count = 1;
  tensor->owning_context = ggml_context_ref (context);
  tensor->tensor = base_tensor;

  return tensor;
}

static GGMLTensor *
ggml_tensor_new_1d (GGMLContext *context, GGMLDataType data_type, size_t size)
{
  return ggml_tensor_from_tensor (context,
                                  ggml_new_tensor_1d (context->ctx,
                                                      (enum ggml_type) data_type,
                                                      size));
}

static GGMLTensor *
ggml_tensor_new_2d (GGMLContext *context, GGMLDataType data_type, size_t width, size_t height)
{
  return ggml_tensor_from_tensor (context,
                                  ggml_new_tensor_2d (context->ctx,
                                                      (enum ggml_type) data_type,
                                                      width,
                                                      height));
}

static GGMLTensor *
ggml_tensor_new_3d (GGMLContext *context, GGMLDataType data_type, size_t width, size_t height, size_t depth)
{
  return ggml_tensor_from_tensor (context,
                                  ggml_new_tensor_3d (context->ctx,
                                                      (enum ggml_type) data_type,
                                                      width,
                                                      height,
                                                      depth));
}

static GGMLTensor *
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
  g_autoptr(GPtrArray) children = g_ptr_array_new_null_terminated (2, (GDestroyNotify) ggml_tensor_unref, TRUE);

  if (tensor->tensor->src0 != NULL)
    {
      /* XXX: This isn't strictly speaking correct -
       * tensor->owning_context->ctx might be different
       * from tensor->src0's context, meaning that if
       * the context is unref'd then tensor->src0's memory
       * goes away. */
      g_ptr_array_add (children,
                       ggml_tensor_from_tensor (tensor->owning_context, tensor->tensor->src0));
    }

if (tensor->tensor->src1 != NULL)
    {
      /* XXX: This isn't strictly speaking correct -
       * tensor->owning_context->ctx might be different
       * from tensor->src0's context, meaning that if
       * the context is unref'd then tensor->src0's memory
       * goes away. */
      g_ptr_array_add (children,
                       ggml_tensor_from_tensor (tensor->owning_context, tensor->tensor->src1));
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

/**
 * ggml_data_type_size: 
 * @data_type: A #GGMLDataType
 *
 * Returns: The size in bytes of this @data_type
 */
size_t
ggml_data_type_size (GGMLDataType data_type)
{
  return ggml_type_size ((enum ggml_type) data_type);
}

G_DEFINE_BOXED_TYPE (GGMLTensor, ggml_tensor, ggml_tensor_ref, ggml_tensor_unref);

static GGMLModel *
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
                    GError **error)
{
  g_autoptr(GGMLComputeGraph) compute_graph = ggml_compute_graph_new (2);
  g_autoptr(GGMLTensor) output = (*model->forward_func) (model,
                                                         hyperparameters,
                                                         inputs,
                                                         forward_parameters,
                                                         compute_graph,
                                                         model->forward_func_user_data,
                                                         error);

  if (output == NULL)
    {
      return NULL;
    }

  ggml_build_forward_expand (&compute_graph->cgraph, output->tensor);
  ggml_graph_compute (output->owning_context->ctx, &compute_graph->cgraph);

  return g_steal_pointer (&output);
}

G_DEFINE_BOXED_TYPE (GGMLModel, ggml_model, ggml_model_ref, ggml_model_unref)

/**
 * ggml_hyperparameters_new:
 * @ordered_keys: (array zero-terminated=1): The hyperparameter names
 * @ordered_values: (array length=n_ordered_values): The hyperparameter values.
 * @n_ordered_values: Number of hyperparameter_values
 */
GGMLHyperparameters *
ggml_hyperparameters_new (const char **ordered_keys, int *ordered_values, size_t n_ordered_values)
{
  GGMLHyperparameters *parameters = g_new0 (GGMLHyperparameters, 1);

  g_assert (g_strv_length ((char **) ordered_keys) == n_ordered_values);

  parameters->ordered_keys = g_strdupv ((char **) ordered_keys);
  parameters->parameters = g_hash_table_new_full (g_str_hash, g_str_equal, NULL, NULL);
  parameters->ref_count = 1;

  const char **ordered_keys_iterator = ordered_keys;
  int *ordered_values_iterator = ordered_values;

  while (*ordered_keys_iterator != NULL)
    {
      g_hash_table_insert (parameters->parameters, (gpointer) *(ordered_keys_iterator++), GINT_TO_POINTER (*(ordered_values_iterator++)));
    }

  return parameters;
}

/**
 * ggml_hyperparameters_get_int32:
 * @hyperparameters: A #GGMLHyperparameters
 * @key: The hyperparameter key to retrieve
 *
 * Gets the corresponding value for @key in the hyperparameters. It is an
 * error to pass an invalid key to this function.
 *
 * Returns: An #int32_t with the corresponding value
 */
int32_t
ggml_hyperparameters_get_int32 (GGMLHyperparameters *hyperparameters, const char *key)
{
  int32_t value;
  gboolean successful = g_hash_table_lookup_extended (hyperparameters->parameters, (gconstpointer) key, NULL, (gpointer *) &value);

  g_assert (successful == TRUE);

  return value;
}

/**
 * ggml_hyperparameters_ref:
 * @hyperparameters: A #GGMLHyperparameters
 *
 * Returns: (transfer full): The #GGMLHyperparameters with the increased ref count.
 */
GGMLHyperparameters *
ggml_hyperparameters_ref (GGMLHyperparameters *hyperparameters)
{
  ++hyperparameters->ref_count;
  return hyperparameters;
}

/**
 * ggml_hyperparameters_unref: (skip)
 * @hyperparameters: A #GGMLHyperparameters
 *
 * Frees the @hyperparameters
 */
void
ggml_hyperparameters_unref (GGMLHyperparameters *hyperparameters)
{
  if (--hyperparameters->ref_count == 0)
    {
      g_clear_pointer (&hyperparameters->parameters, g_hash_table_destroy);
      g_clear_pointer (&hyperparameters->ordered_keys, g_strfreev);
      g_clear_pointer (&hyperparameters, g_free);
    }
}

G_DEFINE_BOXED_TYPE (GGMLHyperparameters, ggml_hyperparameters, ggml_hyperparameters_ref, ggml_hyperparameters_unref)


/**
 * ggml_token_dictionary_new:
 * @tokens: (array zero-terminated=1): The tokens to add to this dictionary, in order
 *
 * Returns: (transfer full): A new #GGMLTokenDictionary
 */
GGMLTokenDictionary *
ggml_token_dictionary_new (const char **tokens)
{
  GGMLTokenDictionary *dictionary = g_new0 (GGMLTokenDictionary, 1);
  dictionary->idx_to_word = g_strdupv ((char **) tokens);
  dictionary->word_to_idx = g_hash_table_new_full (g_str_hash, g_str_equal, NULL, NULL);
  dictionary->ref_count = 1;

  int i = 0;
  const char **tokens_iterator = (const char **) dictionary->idx_to_word;

  while (*tokens_iterator != NULL)
    {
      g_hash_table_insert (dictionary->word_to_idx, (gpointer) *(tokens_iterator++), GINT_TO_POINTER (i++));
    }

  return dictionary;
}

/**
 * ggml_token_dictionary_ref: (skip)
 * @dictionary: A #GGMLTokenDictionary
 *
 * Returns: (transfer full): A #GGMLTokenDictionary
 */
GGMLTokenDictionary *
ggml_token_dictionary_ref (GGMLTokenDictionary *dictionary)
{
  ++dictionary->ref_count;
  return dictionary;
}

/**
 * ggml_token_dictionary_unref: (skip)
 * @dictionary: A #GGMLTokenDictionary
 *
 * Decrements the reference count on @dictionary and frees the underlying
 * dictionary if the reference count goes to zero.
 */
void
ggml_token_dictionary_unref (GGMLTokenDictionary *dictionary)
{
  if (--dictionary->ref_count == 0)
    {
      g_clear_pointer (&dictionary->word_to_idx, g_hash_table_destroy);
      g_clear_pointer (&dictionary->idx_to_word, g_strfreev);
      g_clear_pointer (&dictionary, g_free);
    }
}

/**
 * ggml_token_dictionary_lookup_extended:
 * @token_dictionary: A #GGMLTokenDictionary
 * @key: A key to look up in the @token_dictionary
 * @out_token: (out): The corresponding token
 *
 * Returns: %TRUE if the token was found in the dictionary and @value set
 *          %FALSE otherwise.
 */
gboolean
ggml_token_dictionary_lookup_extended (GGMLTokenDictionary *token_dictionary,
                                       const char *key,
                                       int32_t *out_token)
{
  gpointer lookup_token = NULL;

  if (g_hash_table_lookup_extended (token_dictionary->word_to_idx, key, NULL, (gpointer *) &lookup_token))
    {
      *out_token = GPOINTER_TO_INT (lookup_token);
      return TRUE;
    }

  return FALSE;
}

G_DEFINE_BOXED_TYPE (GGMLTokenDictionary, ggml_token_dictionary, ggml_token_dictionary_ref, ggml_token_dictionary_unref)

#define GPT_SPLIT_REGEX "('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\\s[:alpha:][:digit:]]+|\\s+(?!\\S)|\\s+)"

GPtrArray *
ggml_iterate_words_in_regex (GRegex *regex, const char *string)
{
  g_autoptr(GMatchInfo) match_info = NULL;
  g_autoptr(GPtrArray) words_ptr_array = g_ptr_array_new_full (0, g_free);
  g_regex_match (regex, string, 0, &match_info);
  while (g_match_info_matches (match_info))
    {
      gchar *word = g_match_info_fetch (match_info, 0);
      if (word != NULL)
        {
          g_ptr_array_add (words_ptr_array, word);
        }
      g_match_info_next (match_info, NULL);
    }

  return g_steal_pointer (&words_ptr_array);
}

/**
 * ggml_token_dictionary_decode:
 * @token_dictionary: (transfer none): A #GGMLTokenDictionary
 * @tokens: (array length=n_tokens): An array of #int32_t tokens
 * @n_tokens: Number of tokens in @tokens
 *
 * Decode the token array back into a string. It is an error to
 * pass tokens to this function which are outside the range of tokens
 * in @token_dictionary.
 *
 * Returns: (transfer full) (array zero-terminated=1): A new string with the decoded tokens.
 */
char *
ggml_token_dictionary_decode (GGMLTokenDictionary *token_dictionary,
                              int32_t             *tokens,
                              size_t               n_tokens)
{
  size_t token_dictionary_size = g_hash_table_size (token_dictionary->word_to_idx);
  g_autoptr(GPtrArray) decoded_tokens = g_ptr_array_new_null_terminated (n_tokens, NULL, TRUE);

  for (size_t i = 0; i < n_tokens; ++i)
    {
      g_assert (tokens[i] < token_dictionary_size);

      g_ptr_array_add (decoded_tokens, token_dictionary->idx_to_word[tokens[i]]);
    }

  return g_strjoinv ("", (char **) decoded_tokens->pdata);
}

/**
 * ggml_gpt_tokenize:
 * @token_dictionary: A #GGMLTokenDictionary of tokens
 * @string: A string to tokenize
 * @out_tokens: (out) (array length=out_size): Output tokens from the string
 * @error: A #GError
 *
 * Returns: %TRUE with @out_tokens and @out_size set on success, %FALSE
 *          with @error set otherwise.
 */
gboolean
ggml_gpt_tokenize (GGMLTokenDictionary *token_dictionary,
                   const char *string,
                   int32_t **out_tokens,
                   size_t  *out_size,
                   GError **error)
{
  /* Split first into words */
  g_autoptr(GArray) tokens_array = NULL;
  g_autoptr(GRegex) regex = NULL;
  g_autoptr(GPtrArray) words_ptr_array = NULL;

  regex = g_regex_new (GPT_SPLIT_REGEX,
                       G_REGEX_DEFAULT,
                       G_REGEX_MATCH_DEFAULT,
                       error);

  if (regex == NULL)
    {
      return FALSE;
    }

  words_ptr_array = ggml_iterate_words_in_regex (regex, string);

  /* Now we have to find corresponding tokens in the dictionary */
  tokens_array = g_array_sized_new (FALSE,
                                    TRUE,
                                    sizeof (int32_t),
                                    words_ptr_array->len);

  for (size_t i = 0; i < words_ptr_array->len; ++i)
    {
      const char *word = words_ptr_array->pdata[i];
      size_t word_len = strlen(word);

      for (size_t word_start = 0; word_start < word_len;)
        {
          for (size_t word_end = word_len - 1;
               word_end >= word_start;
               --word_end)
            {
              /* Can't use autofree here because we're in a loop */
              char *candidate = g_strndup (&word[word_start],
                                           word_end - word_start + 1);
              int32_t token = 0;

              if (ggml_token_dictionary_lookup_extended (token_dictionary,
                                                         candidate,
                                                         &token))
                {
                  g_array_append_vals (tokens_array, &token, 1);
                  word_start = word_end + 1;
                  g_free (candidate);
                  break;
                }
              else if (word_end == word_start)
                {
                  ++word_start;
                  g_free (candidate);
                  break;
                }

              g_free (candidate);
            }
        }
    }

  *out_tokens = g_array_steal (tokens_array, out_size);
  return TRUE;
}

/**
 * ggml_language_model_new:
 * @hyperparameters: A #GGMLHyperparameters
 * @dictionary: A #GGMLTokenDictionary
 * @model: A #GGMLModel
 *
 * Creates a new #GGMLLanguageModel
 *
 * Returns: (transfer full): A new #GGMLLanguageModel
 */
GGMLLanguageModel *
ggml_language_model_new (GGMLHyperparameters *hyperparameters, GGMLTokenDictionary *dictionary, GGMLModel *model)
{
  GGMLLanguageModel *language_model = g_new0 (GGMLLanguageModel, 1);
  language_model->hyperparameters = ggml_hyperparameters_ref (hyperparameters);
  language_model->token_dictionary = ggml_token_dictionary_ref (dictionary);
  language_model->model = ggml_model_ref (model);
  language_model->ref_count = 1;

  return language_model;
}

#define GGML_LANGUAGE_MODEL_MAGIC 0x67676d6c

static gboolean
input_stream_read_exactly (GInputStream *istream, char *buffer, size_t read_bytes, GCancellable *cancellable, GError **error)
{
  size_t bytes_read;

  if (!g_input_stream_read_all (istream, buffer, read_bytes, &bytes_read, cancellable, error))
    {
      return FALSE;
    }

  if (bytes_read != read_bytes)
    {
      g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED, "Expected to read %zu bytes but only read %zu bytes, truncated file?", read_bytes, bytes_read);
      return FALSE;
    }

  return TRUE;
}

static gboolean
ggml_language_model_consume_istream_magic (GInputStream *istream, GCancellable *cancellable, GError **error)
{
  uint32_t magic;

  if (!input_stream_read_exactly (istream, (char *) &magic, sizeof (uint32_t), cancellable, error))
    return FALSE;

  if (magic != GGML_LANGUAGE_MODEL_MAGIC)
    {
      g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED, "Invalid magic %#010x expected %#010x", magic, GGML_LANGUAGE_MODEL_MAGIC);
      return FALSE;
    }

  return TRUE;
}

static GGMLHyperparameters *
ggml_load_hyperparameters_from_istream (GInputStream *istream, GCancellable *cancellable, GError **error)
{
  int32_t parameter_values[6];

  if (!input_stream_read_exactly (istream, (char *) parameter_values, sizeof (int32_t) * 6, cancellable, error))
    {
      return NULL;
    }

  const char *parameter_keys[] = {
    "n_vocab",
    "n_ctx",
    "n_embd",
    "n_head",
    "n_layer",
    "ftype",
    NULL
  };

  return ggml_hyperparameters_new (parameter_keys, parameter_values, 6);
}

static GGMLTokenDictionary *
ggml_load_token_dictionary_from_istream (GInputStream *stream, GGMLHyperparameters *hyperparameters, GCancellable *cancellable, GError **error)
{
  int32_t n_vocab = ggml_hyperparameters_get_int32 (hyperparameters, "n_vocab");
  int32_t model_n_vocab_check;

  if (!input_stream_read_exactly (stream, (char *) &model_n_vocab_check, sizeof (int32_t) * 1, cancellable, error))
    {
      return FALSE;
    }

  if (model_n_vocab_check != n_vocab)
    {
      g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED, "Model dictionary n_vocab %d does not match hyperparameters n_vocab %d", model_n_vocab_check, n_vocab);
      return NULL;
    }

  g_autoptr (GPtrArray) words = g_ptr_array_new_full (n_vocab + 1, g_free);

  for (size_t i = 0; i < n_vocab; ++i)
    {
      uint32_t word_size;

      if (!input_stream_read_exactly (stream, (char *) &word_size, sizeof (uint32_t) * 1, cancellable, error))
        {
          return FALSE;
        }

      g_autofree char *buf = g_new0 (char, word_size + 1);

      if (!input_stream_read_exactly (stream, (char *) buf, sizeof (char) * word_size, cancellable, error))
        {
          return FALSE;
        }

      buf[word_size] = '\0';
      g_ptr_array_add (words, g_steal_pointer (&buf));
    }

  g_ptr_array_add (words, NULL);

  /* The strings will be copied into the dictionary and autofree'd from here */
  return ggml_token_dictionary_new ((const char **) words->pdata);
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
  g_autoptr(GPtrArray) loaded_keys = g_ptr_array_new_null_terminated (0, g_free, TRUE);

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

      if (!input_stream_read_exactly (istream, (char *) &name_length, sizeof (int32_t) * 1, cancellable, error))
        {
          return FALSE;
        }

      if (!input_stream_read_exactly (istream, (char *) &ttype, sizeof (int32_t) * 1, cancellable, error))
        {
          return FALSE;
        }

      int dims_buffer[2];

      if (!input_stream_read_exactly (istream, (char *) dims_buffer, sizeof (int32_t) * n_dims, cancellable, error))
        {
          return FALSE;
        }

      int32_t input_stream_tensor_n_elements = product_i32(dims_buffer, n_dims);
      g_autofree char *name_buffer = g_new0 (char, name_length + 1);

      if (!input_stream_read_exactly (istream, (char *) name_buffer, sizeof (char) * name_length, cancellable, error))
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

      size_t bytes_per_element = ggml_data_type_size (ttype);
      size_t allocated_bytes = 0;
      char *tensor_data_ptr = ggml_tensor_get_data (tensor, &allocated_bytes);

      size_t expected_bytes = (tensor_definition_n_elements * bytes_per_element / ggml_tensor_block_size (tensor));
      if (expected_bytes != allocated_bytes)
        {
          g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED, "Tensor %s has allocation of %zu bytes, expected %zu bytes", name_buffer, allocated_bytes, expected_bytes);
          return FALSE;
        }

      /* Now we can read the tensor */
      if (!input_stream_read_exactly (istream, tensor_data_ptr, allocated_bytes, cancellable, error))
        {
          return FALSE;
        }

      g_ptr_array_add (loaded_keys, g_strdup (name_buffer));
    }

  if (out_loaded_keys != NULL)
    {
      *out_loaded_keys = (char **) g_ptr_array_steal (loaded_keys, NULL);
    }

  return TRUE;
}

static GGMLModel *
ggml_load_model_from_istream (GInputStream                           *istream,
                              GGMLModelDescFromHyperparametersFunc    create_model_desc,
                              gpointer                                create_model_desc_user_data,
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
  g_autoptr (GGMLModelDescNode) model_desc_node = (*create_model_desc) (hyperparameters, create_model_desc_user_data);
  g_autoptr (GHashTable) flattened_desc = ggml_model_desc_node_flatten (model_desc_node);
  g_autoptr (GGMLModel) model = ggml_context_new_model_from_flattened_desc (context,
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

static size_t
argmax_f (float *elements, size_t num_elements)
{
  size_t max_idx = 0;
  float max_val = -G_MAXFLOAT;

  for (size_t i = 0; i < num_elements; ++i)
    {
      if (elements[i] > max_val)
        {
          max_idx = i;
          max_val = elements[i];
        }
    }

  return max_idx;
}

static gboolean
ggml_language_model_forward_single_iteration (GGMLModel            *model,
                                              GGMLHyperparameters  *hyperparameters,
                                              GHashTable           *inference_parameters,
                                              int32_t              *input_tokens,
                                              size_t                n_input_tokens,
                                              int32_t              *out_token,
                                              GError              **error)
{
  int32_t n_vocab = ggml_hyperparameters_get_int32 (hyperparameters, "n_vocab");
  g_autoptr(GVariant) variant = g_variant_ref_sink(g_variant_new_fixed_array (G_VARIANT_TYPE_INT32,
                                                                              input_tokens,
                                                                              n_input_tokens,
                                                                              sizeof (int32_t)));
  g_autoptr(GGMLTensor) logits_tensor = ggml_model_forward (model,
                                                            hyperparameters,
                                                            variant,
                                                            inference_parameters,
                                                            error);

  if (logits_tensor == NULL)
    {
      *out_token = -1;
      return FALSE;
    }

  size_t logits_tensor_n_bytes;
  float *logits_tensor_data = (float *) ggml_tensor_get_data (logits_tensor, &logits_tensor_n_bytes);
  float *end_logit_data = logits_tensor_data + (((int32_t) (n_input_tokens - 1)) * n_vocab);
  *out_token = argmax_f (end_logit_data, n_vocab);

  return TRUE;
}

static const char n_past_key[] = "n_past";

static int32_t *
ggml_language_model_forward_loop (GGMLModel           *model,
                                  GGMLHyperparameters *hyperparameters,
                                  int32_t             *initial_prompt_tokens,
                                  size_t               n_initial_prompt_tokens,
                                  int32_t              num_iterations,
                                  size_t              *out_num_tokens,
                                  GError             **error)
{
  /* Assming for now that num_iterations is positive */
  g_autoptr(GArray) prompt_tokens = g_array_sized_new (FALSE, TRUE, sizeof (int32_t), n_initial_prompt_tokens + num_iterations);
  g_autoptr(GHashTable) inference_parameters = g_hash_table_new_full (g_str_hash, g_str_equal, NULL , NULL);

  g_hash_table_insert (inference_parameters, n_past_key, GINT_TO_POINTER (0));

  memcpy (prompt_tokens->data, initial_prompt_tokens, sizeof (int32_t) * n_initial_prompt_tokens);
  prompt_tokens->len = n_initial_prompt_tokens;

  /* Edge case */
  if (num_iterations == 0)
    {
      return g_array_steal (prompt_tokens, out_num_tokens);
    }

  /* We first do a single iteration to populate the key/value memories */
  int32_t argmax;
  if (!ggml_language_model_forward_single_iteration (model,
                                                     hyperparameters,
                                                     inference_parameters,
                                                     (int32_t *) prompt_tokens->data,
                                                     prompt_tokens->len,
                                                     &argmax,
                                                     error))
    {
      return NULL;
    }

  g_array_append_vals (prompt_tokens, &argmax, 1);

  /* Now we have the key/value memories and we can do the inference as usual.
   *
   * Here we pass in one token at a time, eg, the length of the input is always 1
   * and we are using the most recent token. The keys/values from previous iterations
   * are cached. This means that decoding performance can be linear, as opposed
   * to quadratic. */
  for (int32_t i = 0; i < num_iterations - 1; ++i)
    {
      g_hash_table_insert (inference_parameters, (gpointer) n_past_key, GINT_TO_POINTER (n_initial_prompt_tokens + i));

      if (!ggml_language_model_forward_single_iteration (model,
                                                         hyperparameters,
                                                         inference_parameters,
                                                         ((int32_t *) prompt_tokens->data) + n_initial_prompt_tokens + i,
                                                         1,
                                                         &argmax,
                                                         error))
        {
          return NULL;
        }

      g_array_append_vals (prompt_tokens, &argmax, 1);
    }

  *out_num_tokens = 0;

  return g_array_steal (prompt_tokens, out_num_tokens);
}

/**
 * ggml_language_model_complete:
 * @language_model: A #GGMLLanguageModel
 * @prompt: An input prompt
 * @num_iterations: Number of tokens to generate.
 * @error: A #GError
 *
 * Returns: (transfer full): The completed prompt, after running the autoregressive
 *          generation procedure for @num_iterations.
 */
char *
ggml_language_model_complete (GGMLLanguageModel  *language_model,
                              const char         *prompt,
                              int32_t             num_iterations,
                              GError            **error)
{
  g_autofree int32_t *tokens = NULL;
  size_t   n_tokens = 0;

  if (!ggml_gpt_tokenize (language_model->token_dictionary, prompt, &tokens, &n_tokens, error))
    {
      return NULL;
    }

  size_t out_num_tokens = 0;
  g_autofree int32_t *completed_tokens = ggml_language_model_forward_loop (language_model->model,
                                                                           language_model->hyperparameters,
                                                                           tokens,
                                                                           n_tokens,
                                                                           num_iterations,
                                                                           &out_num_tokens,
                                                                           error);

  if (completed_tokens == NULL)
    {
      return NULL;
    }

  return ggml_token_dictionary_decode (language_model->token_dictionary,
                                       completed_tokens,
                                       out_num_tokens);
}

static void
ggml_model_set_possible_tied_weights (GGMLModel *model,
                                      const char **loaded_keys,
                                      const char **src_weights,
                                      const char **dst_weights)
{
  const char **src_weights_it = src_weights;
  const char **dst_weights_it = dst_weights;

  for (; *src_weights_it != NULL && *dst_weights_it != NULL; ++src_weights_it, ++dst_weights_it)
    {
      if (!g_strv_contains (loaded_keys, *dst_weights_it) && g_strv_contains (loaded_keys, *src_weights_it))
        {
          GGMLTensor *src_tensor = ggml_model_get (model, *src_weights_it);
          GGMLTensor *dst_tensor = ggml_model_get (model, *dst_weights_it);

          g_assert (src_tensor != NULL);
          g_assert (dst_tensor != NULL);

          size_t src_n_bytes = 0;
          char *src_data = ggml_tensor_get_data (src_tensor, &src_n_bytes);
          ggml_tensor_set_data (dst_tensor, src_data, src_n_bytes);
        }
    }
}

/**
 * ggml_language_model_load_from_istream:
 * @istream: (transfer none): A #GInputStream
 * @create_model_desc: (transfer none) (scope call): A #GGMLModelDescFromHyperparametersFunc to specify the model structure and weights
 * @create_model_desc_user_data: (closure create_model_desc): A closure for @create_model_desc
 * @forward_func: (scope notified) (nullable): A #GGMLModelFowardFunc
 * @forward_func_user_data: (closure forward_func) (transfer full): The user data for @forward_func
 * @forward_func_user_data_destroy: (destroy forward_func): A #GDestroyNotify for forward_func
 * @cancellable: (transfer none) (nullable): A #GCancellable
 * @error: (nullable): A #GError
 *
 * Reads a GGML language model from @istream , which includes the hyperparameters, token
 * dictionary and model weights and returns a #GGMLLanguageModel
 *
 * Returns: (transfer full): A #GGMLLanguageModel with the loaded weights on success
 */
GGMLLanguageModel *
ggml_language_model_load_from_istream (GInputStream *istream,
                                       GGMLModelDescFromHyperparametersFunc create_model_desc,
                                       gpointer create_model_desc_user_data,
                                       GGMLModelForwardFunc forward_func,
                                       gpointer forward_func_user_data,
                                       GDestroyNotify forward_func_user_data_destroy,
                                       GCancellable *cancellable,
                                       GError **error)
{
  if (!ggml_language_model_consume_istream_magic (istream, cancellable, error))
    {
      return NULL;
    }

  g_autoptr(GGMLHyperparameters) hyperparameters = ggml_load_hyperparameters_from_istream (istream, cancellable, error);

  if (hyperparameters == NULL)
    {
      return NULL;
    }

  g_autoptr(GGMLTokenDictionary) token_dictionary = ggml_load_token_dictionary_from_istream (istream, hyperparameters, cancellable, error);

  if (token_dictionary == NULL)
    {
      return NULL;
    }

  g_autofree char **loaded_keys = NULL;
  g_autoptr(GGMLModel) model = ggml_load_model_from_istream (istream,
                                                             create_model_desc,
                                                             create_model_desc_user_data,
                                                             hyperparameters,
                                                             forward_func,
                                                             forward_func_user_data,
                                                             forward_func_user_data_destroy,
                                                             &loaded_keys,
                                                             cancellable,
                                                             error);

  const char *src_weights[] = {"model/wte", NULL};
  const char *dst_weights[] = {"model/lm_head", NULL};
  ggml_model_set_possible_tied_weights (model, (const char **) loaded_keys, src_weights, dst_weights);

  if (model == NULL)
    {
      return NULL;
    }

  return ggml_language_model_new (hyperparameters,
                                  token_dictionary,
                                  model);
}

/**
 * ggml_language_model_ref:
 * @language_model: A #GGMLLanguageModel
 *
 * Increments the ref count on @language_model
 *
 * Returns: (transfer full): A #GGMLLanguageModel
 */
GGMLLanguageModel *
ggml_language_model_ref (GGMLLanguageModel *language_model)
{
  ++language_model->ref_count;

  return language_model;
}

/**
 * ggml_language_model_unref:
 * @language_model: A #GGMLLanguageModel
 *
 * Decreases the ref count on @language_model . If the ref count goes to
 * zero, then the language model will be cleaned up. Note that the underlying
 * memory is not freed until the corresponding #GGMLContext in the #GGMLModel
 * is released and its memory pool is cleaned up.
 */
void
ggml_language_model_unref (GGMLLanguageModel *language_model)
{
  if (--language_model->ref_count == 0)
    {
      g_clear_pointer (&language_model->hyperparameters, ggml_hyperparameters_unref);
      g_clear_pointer (&language_model->token_dictionary, ggml_token_dictionary_unref);
      g_clear_pointer (&language_model->model, ggml_model_unref);
      g_clear_pointer (&language_model, g_free);
    }
}

G_DEFINE_BOXED_TYPE (GGMLLanguageModel, ggml_language_model, ggml_language_model_ref, ggml_language_model_unref);

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
  GGMLContext *context = g_new0 (GGMLContext, 1);
  struct ggml_init_params params = {
    .mem_size = memory_size,
    .mem_buffer = NULL,
    .no_alloc = FALSE,
  };

  context->ctx = ggml_init (params);
  context->ref_count = 1;

  g_assert (context->ctx != NULL);

  return context;
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
      g_clear_pointer (&context, g_free);
    }
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
ggml_context_new_model_from_flattened_desc (GGMLContext *context,
                                            GHashTable *flattened_desc,
                                            GGMLModelForwardFunc forward_func,
                                            gpointer forward_func_user_data,
                                            GDestroyNotify forward_func_user_data_destroy)
{
  return ggml_model_new_from_flattened_desc (context,
                                             flattened_desc,
                                             forward_func,
                                             forward_func_user_data,
                                             forward_func_user_data_destroy);
}

G_DEFINE_BOXED_TYPE (GGMLContext, ggml_context, ggml_context_ref, ggml_context_unref);

/**
 * ggml_compute_graph_new:
 * @n_threads: Number of threads to use for computation.
 *
 * Returns: (transfer full): A new #GGMLComputeGraph
 */
GGMLComputeGraph *
ggml_compute_graph_new (size_t n_threads)
{
  GGMLComputeGraph *compute_graph = g_new0 (GGMLComputeGraph, 1);
  compute_graph->cgraph.n_threads = n_threads;
  compute_graph->ref_count = 1;

  return compute_graph;
}

/**
 * ggml_compute_graph_ref:
 * @compute_graph: A #GGMLComputeGraph
 *
 * Increase the reference count on @compute_graph
 *
 * Returns: (transfer full): The #GGMLComputeGraph
 */
GGMLComputeGraph *
ggml_compute_graph_ref (GGMLComputeGraph *compute_graph)
{
  ++compute_graph->ref_count;
  return compute_graph;
}

/**
 * ggml_compute_graph_unref:
 * @compute_graph: A #GGMLComputeGraph
 *
 * Decrease the reference count on @compute_graph . If it drops to zero
 * then @compute_graph will be freed.
 */
void
ggml_compute_graph_unref (GGMLComputeGraph *compute_graph)
{
  if (--compute_graph->ref_count)
    {
      g_clear_pointer (&compute_graph, g_free);
    }
}

/**
 * ggml_compute_graph_build_forward_expand:
 * @compute_graph: A #GGMLComputeGraph
 * @tensor: A #GGMLTensor with the end result of the computation
 *
 * Builds the internal compute graph representation based on the end result
 * tensor @tensor .
 */
void
ggml_compute_graph_build_forward_expand (GGMLComputeGraph *compute_graph, GGMLTensor *tensor)
{
  ggml_build_forward_expand (&compute_graph->cgraph, tensor->tensor);
}

/**
 * ggml_compute_graph_compute:
 * @compute_graph: A #GGMLComputeGraph
 * @context: A #GGMLContext used for the computation itself
 *
 * Runs the computation over the compute graph, starting from the input
 * tensors in the computation all the way to the output. After running this,
 * the tensor passed in ggml_compute_graph_build_forward_expand and all of its ancestors
 * will have some defined value.
 */
void
ggml_compute_graph_compute (GGMLComputeGraph *compute_graph,
                            GGMLContext *context)
{
  ggml_graph_compute (context->ctx, &compute_graph->cgraph);
}

G_DEFINE_BOXED_TYPE (GGMLComputeGraph,
                     ggml_compute_graph,
                     ggml_compute_graph_ref,
                     ggml_compute_graph_unref);