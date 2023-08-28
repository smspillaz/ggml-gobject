/*
 * ggml-gobject/ggml-gpt.h
 *
 * Header file for ggml-gpt
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
#include <ggml-gobject/ggml-compute-graph.h>
#include <ggml-gobject/ggml-execution-memory.h>
#include <ggml-gobject/ggml-hyperparameters.h>
#include <ggml-gobject/ggml-language-model.h>
#include <ggml-gobject/ggml-model.h>
#include <ggml-gobject/ggml-model-desc.h>
#include <ggml-gobject/ggml-ops.h>
#include <ggml-gobject/ggml-token-dictionary.h>

G_BEGIN_DECLS

gboolean ggml_gpt_tokenize (GGMLTokenDictionary *token_dictionary,
                            const char *string,
                            int32_t **out_tokens,
                            size_t  *out_size,
                            GError **error);
GGMLTensor * ggml_gpt_model_forward_pass (GGMLModel *model,
                                          GGMLHyperparameters *hyperparameters,
                                          GVariant *inputs,
                                          GHashTable *input_parameters,
                                          GGMLComputeGraph *cgraph,
                                          GGMLExecutionMemory *execution_memory,
                                          gpointer user_data,
                                          GError **error);
GGMLModelDescNode * ggml_create_gpt2_model_desc (int32_t n_vocab,
                                                 int32_t d_model,
                                                 int32_t d_ff,
                                                 int32_t n_layer,
                                                 int32_t n_ctx);

GGMLLanguageModelDesc * ggml_create_gpt2_model_desc_from_hyperparameters (GGMLHyperparameters  *hyperparameters);
const char ** ggml_gpt_model_quantization_regexes (void);

size_t ggml_gpt_model_forward_pass_estimate_memory_buffer_size (size_t n_tokens);
GBytes * ggml_gpt_model_forward_pass_create_memory_buffer (size_t n_tokens);

G_END_DECLS