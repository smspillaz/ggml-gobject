/*
 * ggml-gobject/ggml-language-model.h
 *
 * Header file for ggml-language-model
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
#include <ggml-gobject/ggml-cached-model.h>
#include <ggml-gobject/ggml-hyperparameters.h>
#include <ggml-gobject/ggml-model-desc.h>
#include <ggml-gobject/ggml-model-config.h>
#include <ggml-gobject/ggml-model.h>
#include <ggml-gobject/ggml-token-dictionary.h>

G_BEGIN_DECLS

typedef struct _GGMLLanguageModelCompletionCursor GGMLLanguageModelCompletionCursor;

#define GGML_TYPE_LANGUAGE_MODEL_COMPLETION_CURSOR (ggml_language_model_completion_cursor_get_type ())
GType ggml_language_model_completion_cursor_get_type (void);

GGMLLanguageModelCompletionCursor * ggml_language_model_completion_cursor_ref (GGMLLanguageModelCompletionCursor *cursor);
void ggml_language_model_completion_cursor_unref (GGMLLanguageModelCompletionCursor *cursor);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLLanguageModelCompletionCursor, ggml_language_model_completion_cursor_unref)

typedef struct _GGMLLanguageModel GGMLLanguageModel;

#define GGML_TYPE_LANGUAGE_MODEL (ggml_language_model_get_type ())
GType ggml_language_model_get_type (void);

GGMLLanguageModel *
ggml_language_model_new (GGMLHyperparameters *hyperparameters,
                         GGMLTokenDictionary *dictionary,
                         GGMLModel           *model,
                         GGMLModelDescNode   *memory_desc_node);
GGMLLanguageModel *ggml_language_model_ref (GGMLLanguageModel *language_model);
void ggml_language_model_unref (GGMLLanguageModel *language_model);

typedef struct {
  GGMLModelDescNode *weights_desc;
  GGMLModelDescNode *memory_desc;
} GGMLLanguageModelDesc;

#define GGML_TYPE_LANGUAGE_MODEL_DESC (ggml_language_model_desc_get_type ())
GType ggml_language_model_desc_get_type (void);

GGMLLanguageModelDesc * ggml_language_model_desc_new (GGMLModelDescNode *weights_desc_node,
                                                      GGMLModelDescNode *memory_desc_node);

GGMLLanguageModelDesc * ggml_language_model_desc_copy (GGMLLanguageModelDesc *language_model_desc);
void ggml_language_model_desc_free (GGMLLanguageModelDesc *language_model_desc);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLLanguageModelDesc, ggml_language_model_desc_free);

/**
 * GGMLModelDescFromHyperparametersFunc:
 * @param hyperparameters: (transfer none): A #GGMLHyperparameters
 * @param user_data: (transfer none) (closure): A gpointer containing user-specified data
 *
 * Create a new #GGMLLanguageModelDesc from a #GGMLHyperparameters
 *
 * In general you would specify a callback matching this function signature
 * in order to create the model given some hyperparameters read from a file.
 *
 * Returns: (transfer full): A new #GGMLLanguageModelDesc describing the per-model weights
 *          which can be shared across multiple inference instances and also the per-instance
 *          weights which can't be shared
 */
typedef GGMLLanguageModelDesc *(*GGMLModelDescFromHyperparametersFunc) (
    GGMLHyperparameters *hyperparameters,
    gpointer user_data
);

gboolean ggml_language_model_consume_istream_magic (GInputStream *istream,
                                                    GCancellable *cancellable,
                                                    GError **error);
gboolean ggml_language_model_consume_istream_magic_finish (GAsyncResult  *result,
                                                           GError      **error);
void ggml_language_model_consume_istream_magic_async (GInputStream         *istream,
                                                      GCancellable         *cancellable,
                                                      GAsyncReadyCallback   callback,
                                                      gpointer              user_data);

typedef enum {
  GGML_DEFINED_LANGUAGE_MODEL_GPT2P117M,
  GGML_DEFINED_LANGUAGE_MODEL_GPT2P345M,
  GGML_DEFINED_LANGUAGE_MODEL_GPT2P774M,
  GGML_DEFINED_LANGUAGE_MODEL_GPT2P1558M,
} GGMLDefinedLanguageModel;

GGMLLanguageModel *ggml_language_model_load_from_istream (GInputStream *istream,
                                                          GGMLModelConfig *model_config,
                                                          GGMLModelDescFromHyperparametersFunc create_model_desc,
                                                          gpointer create_model_desc_user_data,
                                                          GGMLModelForwardFunc forward_func,
                                                          gpointer forward_func_user_data,
                                                          GDestroyNotify forward_func_user_data_destroy,
                                                          GCancellable *cancellable,
                                                          GError **error);

void ggml_language_model_load_from_istream_async (GInputStream *istream,
                                                  GGMLModelConfig *model_config,
                                                  GGMLModelDescFromHyperparametersFunc create_model_desc,
                                                  gpointer create_model_desc_user_data,
                                                  GDestroyNotify create_model_desc_user_data_destroy,
                                                  GGMLModelForwardFunc forward_func,
                                                  gpointer forward_func_user_data,
                                                  GDestroyNotify forward_func_user_data_destroy,
                                                  GCancellable *cancellable,
                                                  GAsyncReadyCallback callback,
                                                  gpointer user_data);
GGMLLanguageModel * ggml_language_model_load_from_istream_finish (GAsyncResult  *result,
                                                                  GError       **error);

GGMLLanguageModel *ggml_language_model_load_defined_from_istream (GGMLDefinedLanguageModel   model,
                                                                  GInputStream              *istream,
                                                                  GGMLModelConfig           *model_config,
                                                                  GCancellable              *cancellable,
                                                                  GError                   **error);
void ggml_language_model_load_defined_from_istream_async (GGMLDefinedLanguageModel   model,
                                                          GInputStream              *istream,
                                                          GGMLModelConfig           *model_config,
                                                          GCancellable              *cancellable,
                                                          GAsyncReadyCallback        callback,
                                                          gpointer                   user_data);
GGMLLanguageModel *ggml_language_model_load_defined_from_istream_finish (GAsyncResult  *result,
                                                                         GError       **error);

GGMLCachedModelIstream *ggml_language_model_stream_from_cache (GGMLDefinedLanguageModel   defined_model,
                                                               GError                   **error);

char * ggml_language_model_decode_tokens (GGMLLanguageModel *language_model,
                                          int32_t           *tokens,
                                          size_t             length);

GGMLLanguageModelCompletionCursor * ggml_language_model_create_completion (GGMLLanguageModel *language_model,
                                                                           const char        *prompt,
                                                                           size_t             max_completion_tokens);

typedef void (*GGMLLanguageModelCompletionCursorStreamFunc) (const char *decoded,
                                                             gboolean    is_complete_eos,
                                                             gpointer    user_data);

void ggml_language_model_completion_cursor_exec_stream_async (GGMLLanguageModelCompletionCursor           *cursor,
                                                              size_t                                       num_iterations,
                                                              size_t                                       stream_chunk_size,
                                                              GCancellable                                *cancellable,
                                                              GGMLLanguageModelCompletionCursorStreamFunc  stream_func,
                                                              gpointer                                     stream_func_data,
                                                              GDestroyNotify                               stream_func_data_destroy,
                                                              GAsyncReadyCallback                          callback,
                                                              gpointer                                     user_data);

gboolean ggml_language_model_completion_cursor_exec_stream_finish (GGMLLanguageModelCompletionCursor  *cursor,
                                                                   GAsyncResult                       *result,
                                                                   GError                            **error);

/* Simpler API that does not use streaming */
void ggml_language_model_completion_cursor_exec_async (GGMLLanguageModelCompletionCursor           *cursor,
                                                       size_t                                       num_iterations,
                                                       GCancellable                                *cancellable,
                                                       GAsyncReadyCallback                          callback,
                                                       gpointer                                     user_data);
char * ggml_language_model_completion_cursor_exec_finish (GGMLLanguageModelCompletionCursor  *cursor,
                                                          GAsyncResult                       *result,
                                                          gboolean                           *out_is_complete_eos,
                                                          GError                            **error);

/* Execution */
char * ggml_language_model_completion_cursor_exec (GGMLLanguageModelCompletionCursor  *cursor,
                                                   int32_t                             num_iterations,
                                                   GCancellable                       *cancellable,
                                                   gboolean                           *out_is_complete_eos,
                                                   GError                            **error);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLLanguageModel, ggml_language_model_unref)


G_END_DECLS
