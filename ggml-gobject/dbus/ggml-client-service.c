/*
 * ggml-gobject/dbus/ggml-client-service.c
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

#include <gio/gio.h>
#include <gio/gunixinputstream.h>
#include <gio/gunixoutputstream.h>
#include <gio/gunixfdlist.h>
#include <ggml-gobject/dbus/ggml-service-dbus.h>
#include <ggml-gobject/dbus/ggml-client-service.h>
#include <ggml-gobject/dbus/ggml-client-session.h>
#include <ggml-gobject/dbus/internal/ggml-client-session-internal.h>

#define SIMPLE_TYPE_IO_STREAM  (simple_io_stream_get_type ())
#define SIMPLE_IO_STREAM(o)    (G_TYPE_CHECK_INSTANCE_CAST ((o), SIMPLE_TYPE_IO_STREAM, SimpleIOStream))
#define SIMPLE_IS_IO_STREAM(o) (G_TYPE_CHECK_INSTANCE_TYPE ((o), SIMPLE_TYPE_IO_STREAM))

typedef struct
{
  GIOStream parent_instance;
  GInputStream *input_stream;
  GOutputStream *output_stream;
} SimpleIOStream;

typedef struct
{
  GIOStreamClass parent_class;
} SimpleIOStreamClass;

static GType simple_io_stream_get_type (void) G_GNUC_CONST;

G_DEFINE_TYPE (SimpleIOStream, simple_io_stream, G_TYPE_IO_STREAM)

static void
simple_io_stream_finalize (GObject *object)
{
  SimpleIOStream *stream = SIMPLE_IO_STREAM (object);
  g_object_unref (stream->input_stream);
  g_object_unref (stream->output_stream);
  G_OBJECT_CLASS (simple_io_stream_parent_class)->finalize (object);
}

static void
simple_io_stream_init (SimpleIOStream *stream)
{
}

static GInputStream *
simple_io_stream_get_input_stream (GIOStream *_stream)
{
  SimpleIOStream *stream = SIMPLE_IO_STREAM (_stream);
  return stream->input_stream;
}

static GOutputStream *
simple_io_stream_get_output_stream (GIOStream *_stream)
{
  SimpleIOStream *stream = SIMPLE_IO_STREAM (_stream);
  return stream->output_stream;
}

static void
simple_io_stream_class_init (SimpleIOStreamClass *klass)
{
  GObjectClass *gobject_class;
  GIOStreamClass *giostream_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gobject_class->finalize = simple_io_stream_finalize;

  giostream_class = G_IO_STREAM_CLASS (klass);
  giostream_class->get_input_stream  = simple_io_stream_get_input_stream;
  giostream_class->get_output_stream = simple_io_stream_get_output_stream;
}

static GIOStream *
simple_io_stream_new (GInputStream  *input_stream,
                     GOutputStream *output_stream)
{
  SimpleIOStream *stream;
  g_return_val_if_fail (G_IS_INPUT_STREAM (input_stream), NULL);
  g_return_val_if_fail (G_IS_OUTPUT_STREAM (output_stream), NULL);
  stream = SIMPLE_IO_STREAM (g_object_new (SIMPLE_TYPE_IO_STREAM, NULL));
  stream->input_stream = g_object_ref (input_stream);
  stream->output_stream = g_object_ref (output_stream);
  return G_IO_STREAM (stream);
}

typedef struct _GGMLClientService {
  GGMLService      *proxy;
  GDBusConnection  *dbus_connection;
  size_t            ref_count;
} GGMLClientService;

typedef struct {
  GGMLClientService   *client;
  GCancellable        *cancellable;
  GAsyncReadyCallback  callback;
  gpointer             user_data;
} GGMLClientServiceInitState;


/**
 * ggml_client_service_ref: (skip):
 * @client: A #GGMLClientService
 *
 * Increase ref count on this #GGMLClientService
 *
 * Returns: (transfer full): A #GGMLClientService
 */
GGMLClientService *
ggml_client_service_ref (GGMLClientService *client)
{
  ++client->ref_count;
  return client;
}

/**
 * ggml_client_service_unref: (skip):
 * @client: A #GGMLClientService
 *
 * Decrease ref count on this #GGMLClientService . If this
 * ref count drops to zero, then @client is freed.
 */
void
ggml_client_service_unref (GGMLClientService *client)
{
  if (--client->ref_count == 0)
    {
      g_clear_object (&client->dbus_connection);
      g_clear_object (&client->proxy);
      g_clear_pointer (&client, g_free);
    }
}

static GGMLClientServiceInitState *
ggml_client_service_init_state_new (GGMLClientService   *client,
                                    GCancellable        *cancellable,
                                    GAsyncReadyCallback  callback,
                                    gpointer             user_data)
{
  GGMLClientServiceInitState *state = g_new0 (GGMLClientServiceInitState, 1);
  state->client = ggml_client_service_ref (client);
  state->cancellable = cancellable != NULL ? g_object_ref (cancellable) : NULL;
  state->callback = callback;
  state->user_data = user_data;

  return state;
}

static void
ggml_client_service_init_state_free (GGMLClientServiceInitState *state)
{
  g_clear_object (&state->cancellable);
  g_clear_pointer (&state->client, ggml_client_service_unref);
  g_clear_pointer (&state, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLClientServiceInitState, ggml_client_service_init_state_free);

static void
on_created_private_connection (GObject      *source_object,
                               GAsyncResult *result,
                               gpointer      user_data)
{
  g_autoptr(GGMLClientServiceInitState) state = user_data;
  g_autoptr(GError) error = NULL;
  g_autoptr(GDBusConnection) dbus_connection = g_dbus_connection_new_finish (result,
                                                                             &error);
  g_autoptr(GTask) task = g_task_new (NULL, NULL, state->callback, state->user_data);

  if (dbus_connection == NULL)
    {
      g_task_return_error (task, g_steal_pointer (&error));
      return;
    }

  state->client->dbus_connection = g_object_ref (dbus_connection);

  g_task_return_pointer (task,
                         ggml_client_service_ref (state->client),
                         (GDestroyNotify) ggml_client_service_unref);
}

static void
on_call_open_session_reply (GObject      *source_object,
                            GAsyncResult *result,
                            gpointer      user_data)
{
  GGMLService *service_proxy = GGML_SERVICE (source_object);
  g_autoptr(GGMLClientServiceInitState) state = user_data;
  g_autoptr(GError) error = NULL;
  g_autoptr(GUnixFDList) fd_list = NULL;

  if (!ggml_service_call_open_session_finish (service_proxy,
                                              &fd_list,
                                              result,
                                              &error))
    {
      g_autoptr(GTask) task = g_task_new (NULL, NULL, state->callback, state->user_data);
      g_task_return_error (task, g_steal_pointer (&error));
      return;
    }

  int n_fds = 0;
  g_autofree int *fds = g_unix_fd_list_steal_fds (fd_list, &n_fds);

  g_autoptr(GInputStream) input_stream = g_unix_input_stream_new (fds[0], TRUE);
  g_autoptr(GOutputStream) output_stream = g_unix_output_stream_new (fds[1], TRUE);
  g_autoptr(GIOStream) io_stream = simple_io_stream_new (input_stream, output_stream);

  g_dbus_connection_new (io_stream,
                         NULL,
                         G_DBUS_CONNECTION_FLAGS_AUTHENTICATION_CLIENT,
                         NULL,
                         state->cancellable,
                         on_created_private_connection,
                         g_steal_pointer (&state));
}

static void
on_model_service_proxy_ready (GObject      *source_object,
                              GAsyncResult *result,
                              gpointer      user_data)
{
  g_autoptr(GError) error = NULL;
  g_autoptr(GGMLClientServiceInitState) state = user_data;

  g_autoptr(GGMLService) proxy = ggml_service_proxy_new_for_bus_finish (result, &error);
  state->client->proxy = g_steal_pointer (&proxy);

  if (state->client->proxy == NULL)
    {
      g_autoptr(GTask) task = g_task_new (NULL, NULL, state->callback, state->user_data);
      g_task_return_error (task, g_steal_pointer (&error));
      return;
    }

  ggml_service_call_open_session (state->client->proxy,
                                  G_DBUS_CALL_FLAGS_NONE,
                                  -1,
                                  NULL,
                                  state->cancellable,
                                  on_call_open_session_reply,
                                  g_steal_pointer (&state));
}

/**
 * ggml_client_service_new_async:
 * @cancellable: (nullable): A #GCancellable
 * @callback: A #GAsyncReadyCallback
 * @user_data: (closure callback): User data for @callback
 *
 * Create a new #GGMLClientService asynchronously, and call @callback
 * when it is ready. The call can be completed with %ggml_client_service_new_finish
 */
void
ggml_client_service_new_async (GCancellable       *cancellable,
                               GAsyncReadyCallback callback,
                               gpointer            user_data)
{
  GGMLClientService *client = g_new0 (GGMLClientService, 1);
  client->ref_count = 1;

  g_autoptr(GGMLClientServiceInitState) state = ggml_client_service_init_state_new (g_steal_pointer (&client),
                                                                                    cancellable,
                                                                                    callback,
                                                                                    user_data);

  ggml_service_proxy_new_for_bus (
    G_BUS_TYPE_SESSION,
    G_DBUS_PROXY_FLAGS_NONE,
    "org.ggml.Service",
    "/org/ggml/Service",
    cancellable,
    on_model_service_proxy_ready,
    g_steal_pointer (&state)
 );
}

/**
 * ggml_client_service_new_finish:
 * @result: A #GAsyncResult
 * @error: A #GError out-parameter
 *
 * Completes the call to %ggml_client_service_new_async.
 *
 * Returns: (transfer full): A new #GGMLClientService or %NULL with
 *          @error set on failure.
 */
GGMLClientService *
ggml_client_service_new_finish (GAsyncResult  *result,
                                GError       **error)
{
  return g_task_propagate_pointer (G_TASK (result), error);
}

typedef struct {
  GGMLClientService *client;
  GAsyncReadyCallback     callback;
  gpointer                user_data;
} GGMLClientServiceOpenSessionState;

static GGMLClientServiceOpenSessionState *
ggml_client_service_open_session_state_new (GGMLClientService *client,
                                            GAsyncReadyCallback     callback,
                                            gpointer                user_data)
{
  GGMLClientServiceOpenSessionState *state = g_new0 (GGMLClientServiceOpenSessionState, 1);

  state->client = ggml_client_service_ref (client);
  state->callback = callback;
  state->user_data = user_data;

  return state;
}

static void
ggml_client_service_open_session_state_free (GGMLClientServiceOpenSessionState *state)
{
  g_clear_pointer (&state->client, ggml_client_service_unref);
  g_clear_pointer (&state, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLClientServiceOpenSessionState, ggml_client_service_open_session_state_free);

static void
on_session_proxy_ready (GObject      *source_object,
                        GAsyncResult *result,
                        gpointer      user_data)
{
  g_autoptr(GGMLClientServiceOpenSessionState) state = user_data;
  g_autoptr(GError) error = NULL;
  g_autoptr(GGMLSession) session_proxy = ggml_session_proxy_new_finish (result, &error);
  g_autoptr(GTask) task = g_task_new (NULL, NULL, state->callback, state->user_data);

  if (session_proxy == NULL)
    {
      g_error ("Failed to create SessionProxy object: %s", error->message);
      g_task_return_error (task, g_steal_pointer (&error));
      return;
    }

  g_autoptr(GGMLClientSession) session_client = ggml_client_session_new (session_proxy);
  g_task_return_pointer (task, g_steal_pointer (&session_client), (GDestroyNotify) ggml_client_session_unref);
}

/**
 * ggml_client_service_open_session_async:
 * @client: A #GGMLClientService
 * @cancellable: A #GCancellable
 * @callback: A #GAsyncReadyCallback
 * @user_data: (closure callback): User data for @callback
 *
 * Asynchronously open a new session on @client . When the session is ready,
 * @callback will be invoked, you can complete the call with
 * %ggml_client_service_open_new_session_finish
 */
void
ggml_client_service_open_session_async (GGMLClientService   *client,
                                        GCancellable        *cancellable,
                                        GAsyncReadyCallback  callback,
                                        gpointer             user_data)
{
  g_autoptr(GGMLClientServiceOpenSessionState) state = ggml_client_service_open_session_state_new (client, callback, user_data);

  ggml_session_proxy_new (
    client->dbus_connection,
    G_DBUS_PROXY_FLAGS_NONE,
    NULL, /* Not a bus connection */
    "/org/ggml/Session",
    cancellable,
    on_session_proxy_ready,
    g_steal_pointer (&state)
 );
}

/**
 * ggml_client_service_open_new_session_finish:
 * @result: A @GAsyncResult
 * @error: A #GError out-parameter
 *
 * Completes the call to %ggml_client_service_open_new_session_async .
 *
 * Returns: (transfer full): A new #GGMLClientSession or %NULL with
 *          @error set on failure.
 */
GGMLClientSession *
ggml_client_service_open_session_finish (GAsyncResult  *result,
                                         GError       **error)
{
  return g_task_propagate_pointer (G_TASK (result), error);
}

G_DEFINE_BOXED_TYPE (GGMLClientService,
                     ggml_client_service,
                     ggml_client_service_ref,
                     ggml_client_service_unref);
