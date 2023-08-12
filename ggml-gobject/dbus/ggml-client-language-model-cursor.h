/*
 * ggml-gobject/dbus/ggml-client-language-model-cursor.h
 *
 * Copyright (c) 2023 Sam Spilsbury
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

#pragma once

#include <glib-object.h>
#include <gio/gio.h>

G_BEGIN_DECLS

typedef struct _GGMLClientLanguageModelCursor GGMLClientLanguageModelCursor;
#define GGML_CLIENT_TYPE_LANGUAGE_MODEL_CURSOR (ggml_client_language_model_cursor_get_type ())
GType ggml_client_language_model_cursor_get_type (void);

/**
 * GGMLClientLanguageModelCursorChunkCallback:
 * @chunk: A chunk generated by the language model.
 * @user_data: Some user data.
 *
 * A callback type for streaming chunks from the cursor.
 */
typedef void (*GGMLClientLanguageModelCursorChunkCallback) (const char *chunk,
                                                            gpointer    user_data);

GGMLClientLanguageModelCursor * ggml_client_language_model_cursor_ref (GGMLClientLanguageModelCursor *cursor);
void ggml_client_language_model_cursor_unref (GGMLClientLanguageModelCursor *cursor);

void ggml_client_language_model_cursor_exec_async (GGMLClientLanguageModelCursor *cursor,
                                                   size_t                         num_tokens,
                                                   GCancellable                  *cancellable,
                                                   GAsyncReadyCallback            callback,
                                                   gpointer                       user_data);
char * ggml_client_language_model_cursor_exec_finish (GGMLClientLanguageModelCursor  *cursor,
                                                      GAsyncResult                   *result,
                                                      GError                        **error);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLClientLanguageModelCursor, ggml_client_language_model_cursor_unref)

G_END_DECLS
