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
#include <ggml-gobject/ggml-hyperparameters.h>
#include <ggml-gobject/ggml-model.h>
#include <ggml-gobject/ggml-token-dictionary.h>

G_BEGIN_DECLS

typedef struct _GGMLLanguageModel GGMLLanguageModel;

#define GGML_TYPE_LANGUAGE_MODEL (ggml_language_model_get_type ());
GType ggml_language_model_get_type (void);

GGMLLanguageModel *
ggml_language_model_new (GGMLHyperparameters *hyperparameters,
                         GGMLTokenDictionary *dictionary,
                         GGMLModel *model);
GGMLLanguageModel *ggml_language_model_ref (GGMLLanguageModel *language_model);
void ggml_language_model_unref (GGMLLanguageModel *language_model);

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
  GGML_DEFINED_LANGUAGE_MODEL_GPT2
} GGMLDefinedLanguageModel;

GGMLLanguageModel *ggml_language_model_load_from_istream (GInputStream *istream,
                                                          GGMLModelDescFromHyperparametersFunc create_model_desc,
                                                          gpointer create_model_desc_user_data,
                                                          GGMLModelForwardFunc forward_func,
                                                          gpointer forward_func_user_data,
                                                          GDestroyNotify forward_func_user_data_destroy,
                                                          GCancellable *cancellable,
                                                          GError **error);

void ggml_language_model_load_from_istream_async (GInputStream *istream,
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
                                                                  GCancellable              *cancellable,
                                                                  GError                   **error);
void ggml_language_model_load_defined_from_istream_async (GGMLDefinedLanguageModel   model,
                                                          GInputStream              *istream,
                                                          GCancellable              *cancellable,
                                                          GAsyncReadyCallback        callback,
                                                          gpointer                   user_data,
                                                          GError                   **error);
GGMLLanguageModel *ggml_language_model_load_defined_from_istream_finish (GAsyncResult  *result,
                                                                         GError       **error);

GFileInputStream *ggml_language_model_stream_from_cache (GGMLDefinedLanguageModel   defined_model,
                                                         GError                   **error);

char * ggml_language_model_complete (GGMLLanguageModel  *language_model,
                                     const char         *prompt,
                                     int32_t             num_iterations,
                                     GCancellable       *cancellable,
                                     gboolean           *out_is_complete_eos,
                                     GError            **error);

char * ggml_language_model_decode_tokens (GGMLLanguageModel *language_model,
                                          int32_t           *tokens,
                                          size_t             length);
char * ggml_language_model_complete_finish (GGMLLanguageModel  *language_model,
                                            GAsyncResult       *result,
                                            gboolean           *out_is_complete,
                                            gboolean           *out_is_complete_eos,
                                            GError            **error);
GThread * ggml_language_model_complete_async (GGMLLanguageModel    *language_model,
                                         const char           *prompt,
                                         size_t                num_iterations,
                                         size_t                chunk_size,
                                         GCancellable         *cancellable,
                                         GAsyncReadyCallback   callback,
                                         gpointer              user_data,
                                         GDestroyNotify        user_data_destroy);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLLanguageModel, ggml_language_model_unref)


G_END_DECLS