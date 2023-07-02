/*
 * ggml-gobject/ggml-model.h
 *
 * Header file for ggml-model
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
#include <gio/gio.h>
#include <ggml-gobject/ggml-compute-graph.h>
#include <ggml-gobject/ggml-hyperparameters.h>
#include <ggml-gobject/ggml-model-desc.h>
#include <ggml-gobject/ggml-tensor.h>

G_BEGIN_DECLS

typedef struct _GGMLModel GGMLModel;

#define GGML_TYPE_MODEL (ggml_model_get_type ());
GType ggml_model_get_type (void);

/**
 * GGMLModelForwardFunc:
 * @model: (transfer none): A #GGMLModel
 * @hyperparameters: (transfer none): A #GGMLHyperparameters
 * @inputs: (transfer none): A #GVariant with inputs used for the forward computation
 * @input_parameters: (nullable) (element-type utf8 int): A #GHashTable with some parameters for the input
 * @compute_graph: (transfer none): A #GGMLComputeGraph which can be added to.
 * @mem_buffer: (transfer none) (nullable): A #GBytes memory buffer to be re-used.
 * @user_data: (closure): A closure with user data to evaluate the function
 * @error: A #GError out-variable
 *
 * XXX: We would like to be able to return a #GPtrArray of output nodes here and avoid passing
 * the @compute_graph, but it seems that gjs struggles to handle a transfer full #GPtrArray out-param.
 *
 * Returns: (transfer full): A new #GGMLTensor output node, used to define a computation graph for a forward pass,
 *          or %NULL with @error set on failure. Note that this callback doesn't actually compute the end result,
 *          merely just defines the compute graph.
 */
typedef GGMLTensor * (*GGMLModelForwardFunc) (GGMLModel   *model,
                                              GGMLHyperparameters *hyperparameters,
                                              GVariant    *inputs,
                                              GHashTable  *input_parameters,
                                              GGMLComputeGraph *compute_graph,
                                              GBytes      *mem_buffer,
                                              gpointer     user_data,
                                              GError     **error);

GGMLModel *ggml_model_ref (GGMLModel *model);
void ggml_model_unref (GGMLModel *model);
GGMLTensor *ggml_model_get (GGMLModel *model, const char *key);
GGMLTensor *ggml_model_forward (GGMLModel *model,
                                GGMLHyperparameters *hyperparameters,
                                GVariant *inputs,
                                GHashTable *forward_parameters,
                                GBytes   *mem_buffer,
                                GError **error);

GGMLModel * ggml_model_load_from_istream (GInputStream                           *istream,
                                          GGMLModelDescNode                      *model_desc_node,
                                          GGMLHyperparameters                    *hyperparameters,
                                          GGMLModelForwardFunc                    forward_func,
                                          gpointer                                forward_func_user_data,
                                          GDestroyNotify                          forward_func_user_data_destroy,
                                          char                                 ***out_loaded_keys,
                                          GCancellable                           *cancellable,
                                          GError                                **error);
void ggml_model_load_from_istream_async (GInputStream *istream,
                                         GGMLModelDescNode *model_desc,
                                         GGMLHyperparameters *hyperparameters,
                                         GGMLModelForwardFunc forward_func,
                                         gpointer forward_func_user_data,
                                         GDestroyNotify forward_func_user_data_destroy,
                                         GCancellable *cancellable,
                                         GAsyncReadyCallback callback,
                                         gpointer user_data);
GGMLModel * ggml_model_load_from_istream_finish (GAsyncResult  *result,
                                                 char        ***out_loaded_keys,
                                                 GError       **error);


GGMLModel * ggml_model_new_from_flattened_desc (GGMLContext *context,
                                                GHashTable  *flattened_desc,
                                                GGMLModelForwardFunc forward_func,
                                                gpointer forward_func_user_data,
                                                GDestroyNotify forward_func_user_data_destroy);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLModel, ggml_model_unref)

G_END_DECLS