/*
 * ggml-gobject/ggml-model-desc.h
 *
 * Header file for ggml-model-desc
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
#include <ggml-gobject/ggml-hyperparameters.h>
#include <ggml-gobject/ggml-types.h>

G_BEGIN_DECLS

/**
 * GGMLModelDescLeaf:
 * @dimensions: (array length=n_dim): An #int64_t with the dimensions of the
 * tensor at this leaf
 * @n_dim: Number of dimensions in this tensor
 */
typedef struct
{
  int64_t *dimensions;
  size_t n_dim;
  GGMLDataType type;
} GGMLModelDescLeaf;

#define GGML_TYPE_MODEL_DESC_LEAF (ggml_model_desc_leaf_get_type ())

GType ggml_model_desc_leaf_get_type (void);

GGMLModelDescLeaf *ggml_model_desc_leaf_new (int64_t *dimensions, size_t n_dim,
                                             GGMLDataType type);
GGMLModelDescLeaf *ggml_model_desc_leaf_ref (GGMLModelDescLeaf *leaf);
void ggml_model_desc_leaf_unref (GGMLModelDescLeaf *leaf);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLModelDescLeaf, ggml_model_desc_leaf_unref)

/**
 * GGMLModelDescNode:
 * @children: (element-type utf8 GGMLModelDescNode): A #GHashTable with this
 * node's children
 * @leaf: A #GGMLModelDescLeaf
 */
typedef struct
{
  GHashTable *children;
  GGMLModelDescLeaf *leaf;
} GGMLModelDescNode;

#define GGML_TYPE_MODEL_DESC_NODE (ggml_model_desc_node_get_type ());
GType ggml_model_desc_node_get_type (void);

GGMLModelDescNode *ggml_model_desc_node_new (GGMLModelDescLeaf *leaf,
                                             GHashTable *children);
GGMLModelDescNode *ggml_model_desc_node_new_leaf (int64_t *dimensions,
                                                  size_t n_dim,
                                                  GGMLDataType type);
GGMLModelDescNode *ggml_model_desc_node_ref (GGMLModelDescNode *node);
GHashTable *ggml_model_desc_node_flatten (GGMLModelDescNode *node);

/**
 * GGMLModelDescMapFunc:
 * @path: The key for the current weight
 * @leaf: (transfer none): A #GGMLModelDescLeaf
 *
 * Returns: (transfer full): A new #GGMLModelDescLeaf with the mapped leaf node.
 */
typedef GGMLModelDescLeaf * (*GGMLModelDescMapFunc) (const char               *path,
                                                     const GGMLModelDescLeaf  *leaf,
                                                     gpointer                  user_data);

GGMLModelDescNode *ggml_model_desc_map (GGMLModelDescNode    *model_desc,
                                        GGMLModelDescMapFunc  map_func,
                                        gpointer              map_user_data);
void ggml_model_desc_node_unref (GGMLModelDescNode *node);

/**
 * GGMLModelDescFromHyperparametersFunc:
 * @param hyperparameters: (transfer none): A #GGMLHyperparameters
 * @param user_data: (transfer none) (closure): A gpointer containing user-specified data
 *
 * Create a new #GGMLModelDescNode from a #GGMLHyperparameters
 *
 * In general you would specify a callback matching this function signature
 * in order to create the model given some hyperparameters read from a file.
 *
 * Returns: (transfer full): A new #GGMLModelDescNode
 */
typedef GGMLModelDescNode *(*GGMLModelDescFromHyperparametersFunc) (
    GGMLHyperparameters *hyperparameters,
    gpointer user_data);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLModelDescNode, ggml_model_desc_node_unref)

G_END_DECLS