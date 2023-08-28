/*
 * ggml-gobject/dbus/ggml-client-service.h
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
#include <ggml-gobject/dbus/ggml-client-session.h>

G_BEGIN_DECLS

typedef struct _GGMLClientService GGMLClientService;

#define GGML_CLIENT_TYPE_SERVICE (ggml_client_service_get_type ())
GType ggml_client_service_get_type (void);

void ggml_client_service_new_async (GCancellable       *cancellable,
                                    GAsyncReadyCallback callback,
                                    gpointer            user_data);
GGMLClientService * ggml_client_service_new_finish (GAsyncResult  *result,
                                                    GError       **error);
GGMLClientService * ggml_client_service_ref (GGMLClientService *client);
void ggml_client_service_unref (GGMLClientService *client);

void ggml_client_service_open_session_async (GGMLClientService   *client,
                                             GCancellable        *cancellable,
                                             GAsyncReadyCallback  callback,
                                             gpointer             user_data);
GGMLClientSession * ggml_client_service_open_session_finish (GAsyncResult  *result,
                                                             GError       **error);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLClientService, ggml_client_service_unref)

G_END_DECLS
