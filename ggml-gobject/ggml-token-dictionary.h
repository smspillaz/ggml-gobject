/*
 * ggml-gobject/ggml-token-dictionary.h
 *
 * Header file for ggml-token-dictionary
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

G_BEGIN_DECLS

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

GGMLTokenDictionary * ggml_token_dictionary_load_from_istream (GInputStream *istream,
                                                               int32_t n_vocab,
                                                               GCancellable *cancellable,
                                                               GError **error);
void ggml_token_dictionary_load_from_istream_async (GInputStream *istream,
                                                    int32_t       n_vocab,
                                                    GCancellable *cancellable,
                                                    GAsyncReadyCallback callback,
                                                    gpointer user_data);
GGMLTokenDictionary * ggml_token_dictionary_load_from_istream_finish (GAsyncResult  *result,
                                                                      GError       **error);

char * ggml_token_dictionary_decode (GGMLTokenDictionary *token_dictionary,
                                     int32_t *tokens,
                                     size_t n_tokens);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLTokenDictionary,
                               ggml_token_dictionary_unref)


G_END_DECLS