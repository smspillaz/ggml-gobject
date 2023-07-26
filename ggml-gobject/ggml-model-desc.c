/*
 * ggml-gobject/ggml-model-desc.c
 *
 * Library code for ggml-model-desc
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

#include <ggml-gobject/ggml-model-desc.h>

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

GGMLModelDescNode *
ggml_model_desc_map_recurse (GGMLModelDescNode    *current_node,
                             GGMLModelDescMapFunc  map_func,
                             gpointer              map_user_data,
                             const char           *path)
{
  g_autoptr(GGMLModelDescLeaf) mapped_leaf = NULL;

  if (current_node->leaf != NULL)
    {
      mapped_leaf = map_func (path,
                              current_node->leaf,
                              map_user_data);
    }

  g_autoptr(GHashTable) mapped_children = NULL;

  if (current_node->children)
    {
      GHashTableIter iter;
      gpointer key, value;

      mapped_children = g_hash_table_new_full (g_str_hash,
                                               g_str_equal,
                                               g_free,
                                               (GDestroyNotify) ggml_model_desc_node_unref);

      g_hash_table_iter_init (&iter, current_node->children);
      while (g_hash_table_iter_next (&iter, &key, &value))
        {
          const char *key_str = key;
          GGMLModelDescNode *child = value;

          g_autofree char *child_path = (path == NULL ?
                                         g_strdup(key) :
                                         g_strjoin("/", path, (const gchar *) key, NULL));

          GGMLModelDescNode *new_child = ggml_model_desc_map_recurse (child,
                                                                      map_func,
                                                                      map_user_data,
                                                                      child_path);

          g_hash_table_insert (mapped_children, g_strdup (key_str), g_steal_pointer (&new_child));
        }
    }

  return ggml_model_desc_node_new (g_steal_pointer (&mapped_leaf),
                                   g_steal_pointer (&mapped_children));
}

/**
 * ggml_model_desc_map:
 * @model_desc: A #GGMLModelDescNode to apply @map_func to
 * @map_func: (scope call): A #GGMLModelDescMapFunc
 * @map_user_data: (closure map_func): User data for @map_func
 *
 * Map @model_desc and return a new #GGMLModelDescNode tree with
 * the @map_func applied to each leaf. The @map_func must return a
 * new #GGMLModelDescLeaf for each leaf.
 *
 * Returns: (transfer full): A new #GGMLModelDescNode transformed by @map_func.
 */
GGMLModelDescNode *
ggml_model_desc_map (GGMLModelDescNode    *model_desc,
                     GGMLModelDescMapFunc  map_func,
                     gpointer              map_user_data)
{
  return ggml_model_desc_map_recurse (model_desc,
                                      map_func,
                                      map_user_data,
                                      NULL);
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
