/*
 * ggml-gobject/dbus/ggml-client-session.h
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
#include <ggml-gobject/dbus/ggml-client-language-model-cursor.h>

G_BEGIN_DECLS

typedef struct _GGMLClientSession GGMLClientSession;

#define GGML_CLIENT_TYPE_SESSION (ggml_client_session_get_type ())
GType ggml_client_session_get_type (void);

GGMLClientSession * ggml_client_session_ref (GGMLClientSession *session_client);
void ggml_client_session_unref (GGMLClientSession *session_client);

void ggml_client_session_new_async (GCancellable        *cancellable,
                                    GAsyncReadyCallback  callback,
                                    gpointer             user_data);
GGMLClientSession * ggml_client_session_new_finish (GAsyncResult  *result,
                                                    GError       **error);

void ggml_client_session_start_completion_async (GGMLClientSession                          *session_client,
                                                 const char                                 *model_name,
                                                 const char                                 *model_variant,
                                                 const char                                 *quantization,
                                                 const char                                 *prompt,
                                                 size_t                                      max_tokens,
                                                 GCancellable                               *cancellable,
                                                 GAsyncReadyCallback                         callback,
                                                 gpointer                                    user_data);
GGMLClientLanguageModelCursor * ggml_client_session_start_completion_finish (GAsyncResult  *result,
                                                                             GError       **error);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLClientSession, ggml_client_session_unref)

G_END_DECLS
