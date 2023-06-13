/*
 * ggml-gobject/ggml-gobject.h
 *
 * Header file for ggml-gobject
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

#include <ggml.h>
#include <gio/gio.h>
#include <glib-object.h>

G_BEGIN_DECLS

typedef enum
{
  GGML_DATA_TYPE_F32 = GGML_TYPE_F32,
  GGML_DATA_TYPE_F16 = GGML_TYPE_F16,
  GGML_DATA_TYPE_Q4_0 = GGML_TYPE_Q4_0,
  GGML_DATA_TYPE_Q4_1 = GGML_TYPE_Q4_1,
  GGML_DATA_TYPE_Q5_0 = GGML_TYPE_Q5_0,
  GGML_DATA_TYPE_Q5_1 = GGML_TYPE_Q5_1,
  GGML_DATA_TYPE_Q8_0 = GGML_TYPE_Q8_0,
  GGML_DATA_TYPE_Q8_1 = GGML_TYPE_Q8_1,
  GGML_DATA_TYPE_I8 = GGML_TYPE_I8,
  GGML_DATA_TYPE_I16 = GGML_TYPE_I16,
  GGML_DATA_TYPE_I32 = GGML_TYPE_I32,
} GGMLDataType;

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
void ggml_model_desc_node_unref (GGMLModelDescNode *node);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLModelDescNode, ggml_model_desc_node_unref)

typedef struct _GGMLTensor GGMLTensor;

#define GGML_TYPE_TENSOR (ggml_tensor_get_type ())
GType ggml_tensor_get_type (void);

GGMLTensor *ggml_tensor_ref (GGMLTensor *tensor);
void ggml_tensor_unref (GGMLTensor *tensor);
size_t ggml_tensor_element_size (GGMLTensor *tensor);
size_t ggml_tensor_n_elements (GGMLTensor *tensor);
size_t ggml_tensor_block_size (GGMLTensor *tensor);
size_t ggml_tensor_n_bytes (GGMLTensor *tensor);
void ggml_tensor_set_data (GGMLTensor *tensor, char *data, size_t size);
void ggml_tensor_set_data_from_bytes (GGMLTensor *tensor, GBytes *bytes);
void ggml_tensor_set_data_from_int32_array (GGMLTensor *tensor,
                                            int32_t    *array,
                                            size_t      n_elements);
char * ggml_tensor_get_data (GGMLTensor *tensor, size_t *out_n_bytes);
GBytes * ggml_tensor_get_bytes (GGMLTensor *tensor);

void ggml_tensor_set_name (GGMLTensor *tensor,
                           const char *name);
const char * ggml_tensor_get_name (GGMLTensor *tensor);

int64_t *ggml_tensor_get_shape (GGMLTensor *tensor, size_t *out_n_dims);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLTensor, ggml_tensor_unref)

size_t ggml_data_type_size (GGMLDataType data_type);

typedef struct _GGMLHyperparameters GGMLHyperparameters;

#define GGML_TYPE_HYPERPARAMETERS (ggml_hyperparameters_get_type ());
GType ggml_hyperparameters_get_type (void);

GGMLHyperparameters *ggml_hyperparameters_new (const char **ordered_keys,
                                               int *ordered_values,
                                               size_t n_ordered_values);
int32_t ggml_hyperparameters_get_int32 (GGMLHyperparameters *hyperparameters,
                                        const char *key);
GGMLHyperparameters *
ggml_hyperparameters_ref (GGMLHyperparameters *hyperparameters);
void ggml_hyperparameters_unref (GGMLHyperparameters *hyperparameters);

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

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLHyperparameters, ggml_hyperparameters_unref)

typedef struct _GGMLModel GGMLModel;

#define GGML_TYPE_MODEL (ggml_model_get_type ());
GType ggml_model_get_type (void);

GGMLModel *ggml_model_ref (GGMLModel *model);
void ggml_model_unref (GGMLModel *model);
GGMLTensor *ggml_model_get (GGMLModel *model, const char *key);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLModel, ggml_model_unref)

typedef struct _GGMLTokenDictionary GGMLTokenDictionary;

#define GGML_TYPE_TOKEN_DICTIONARY (ggml_token_dictionary_get_type ());
GType ggml_token_dictionary_get_type (void);

GGMLTokenDictionary *ggml_token_dictionary_new (const char **tokens);
GGMLTokenDictionary *
ggml_token_dictionary_ref (GGMLTokenDictionary *dictionary);
void ggml_token_dictionary_unref (GGMLTokenDictionary *dictionary);
gboolean ggml_token_dictionary_lookup_extended (GGMLTokenDictionary *token_dictionary,
                                                const char *key,
                                                int32_t *out_token);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLTokenDictionary,
                               ggml_token_dictionary_unref)

typedef struct _GGMLLanguageModel GGMLLanguageModel;

#define GGML_TYPE_LANGUAGE_MODEL (ggml_language_model_get_type ());
GType ggml_language_model_get_type (void);

GGMLLanguageModel *
ggml_language_model_new (GGMLHyperparameters *hyperparameters,
                         GGMLTokenDictionary *dictionary, GGMLModel *model);
GGMLLanguageModel *ggml_language_model_ref (GGMLLanguageModel *language_model);
void ggml_language_model_unref (GGMLLanguageModel *language_model);
GGMLLanguageModel *ggml_language_model_load_from_istream (
    GInputStream *istream,
    GGMLModelDescFromHyperparametersFunc create_model_desc,
    GCancellable *cancellable, GError **error);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLLanguageModel, ggml_language_model_unref)

typedef struct _GGMLContext GGMLContext;

#define GGML_TYPE_CONTEXT (ggml_context_get_type ())
GType ggml_context_get_type (void);

GGMLContext *ggml_context_new (size_t memory_size);
GGMLContext *ggml_context_ref (GGMLContext *context);
void ggml_context_unref (GGMLContext *context);
GGMLTensor *ggml_context_new_tensor_1d (GGMLContext *context,
                                        GGMLDataType data_type, size_t size);
GGMLTensor *ggml_context_new_tensor_2d (GGMLContext *context,
                                        GGMLDataType data_type, size_t width,
                                        size_t height);
GGMLTensor *ggml_context_new_tensor_3d (GGMLContext *context,
                                        GGMLDataType data_type, size_t width,
                                        size_t height, size_t depth);
GGMLModel *
ggml_context_new_model_from_flattened_desc (GGMLContext *context,
                                            GHashTable *flattened_desc);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLContext, ggml_context_unref)


gboolean ggml_gpt_tokenize (GGMLTokenDictionary *token_dictionary,
                            const char *string,
                            int32_t **out_tokens,
                            size_t  *out_size,
                            GError **error);

G_END_DECLS