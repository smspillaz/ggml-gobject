/*
 * examples/service-client/client.c
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

#include <stdio.h>
#include <glib-object.h>
#include <glib-unix.h>
#include <gio/gio.h>
#include <gio/gunixinputstream.h>
#include <gio/gunixoutputstream.h>
#include <ggml-gobject/ggml-language-model-service-dbus.h>

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

typedef struct {
  GMainLoop                       *loop;
  GGMLLanguageModelService        *service_proxy;
  GDBusConnection                 *dbus_connection;
  GGMLLanguageModelServiceSession *session_proxy;
  size_t                           ref_count;
} GGMLLanguageModelClientState;

static GGMLLanguageModelClientState *
ggml_language_model_client_state_new (GMainLoop *loop)
{
  GGMLLanguageModelClientState *state = g_new0 (GGMLLanguageModelClientState, 1);
  state->loop = g_main_loop_ref (loop);
  state->dbus_connection = NULL;
  state->service_proxy = NULL;
  state->session_proxy = NULL;
  state->ref_count = 1;

  return state;
}

static GGMLLanguageModelClientState *
ggml_language_model_client_state_ref (GGMLLanguageModelClientState *state)
{
  ++state->ref_count;
  return state;
}

static void
ggml_language_model_client_state_unref (GGMLLanguageModelClientState *state)
{
  if (--state->ref_count == 0)
    {
      g_message ("Closing client state");
      g_clear_object (&state->session_proxy);
      g_clear_object (&state->dbus_connection);
      g_clear_object (&state->service_proxy);
      g_clear_pointer (&state->loop, g_main_loop_unref);
      g_clear_pointer (&state, g_free);
    }
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLLanguageModelClientState, ggml_language_model_client_state_unref)

static void
on_new_chunk_from_completion (GGMLLanguageModelCompletion *completion,
                              const char *chunk)
{
  printf ("%s", chunk);
  fflush (stdout);
}

static void
on_done_complete_exec (GObject      *source_object,
                       GAsyncResult *result,
                       gpointer      user_data)
{
  g_autoptr(GError) error = NULL;
  g_autoptr(GGMLLanguageModelClientState) state = user_data;
  g_autofree char *completed_string = NULL;

  printf("\n");

  if (!ggml_language_model_completion_call_exec_finish (GGML_LANGUAGE_MODEL_COMPLETION (source_object),
                                                        &completed_string,
                                                        result,
                                                        &error))
    {
      g_error ("Error when calling completion.exec(): %s", error->message);
      g_main_loop_quit (state->loop);
      return;
    }

  g_message ("Completion done");
  g_main_loop_quit (state->loop);
}

static void
on_completion_object_ready (GObject      *source_object,
                            GAsyncResult *result,
                            gpointer      user_data)
{
  g_autoptr(GError) error = NULL;
  g_autoptr(GGMLLanguageModelClientState) state = user_data;
  g_autoptr(GGMLLanguageModelCompletion) completion_proxy = ggml_language_model_completion_proxy_new_finish (result, &error);

  if (completion_proxy == NULL)
    {
      g_error ("Failed to create LanguageModelCompletion proxy: %s", error->message);
      g_main_loop_quit (state->loop);
      return;
    }

  g_message ("Created completion proxy");

  /* Now connect to the new-chunk signal and also call
   * exec() on the proxy. We can interactively start printing the result
   * on the console */
  g_signal_connect (completion_proxy,
                    "new-chunk",
                    G_CALLBACK (on_new_chunk_from_completion),
                    NULL);

  ggml_language_model_completion_call_exec (completion_proxy,
                                            128,
                                            G_DBUS_CALL_FLAGS_NONE,
                                            -1,
                                            NULL,
                                            on_done_complete_exec,
                                            g_steal_pointer (&state));
}

static void
on_call_create_completion_reply (GObject      *source_object,
                                 GAsyncResult *result,
                                 gpointer      user_data)
{
  g_autoptr(GGMLLanguageModelClientState) state = user_data;
  g_autoptr(GError) error = NULL;
  g_autofree char *completion_object_path = NULL;

  if (!ggml_language_model_service_session_call_create_completion_finish (state->session_proxy,
                                                                          &completion_object_path,
                                                                          result,
                                                                          &error))
    {
      g_error ("Failed to create completion: %s\n", error->message);
      g_main_loop_quit (state->loop);
      return;
    }

  /* Now we have to create a proxy for the object path on the bus */
  g_message ("Created completion on server side, creating object for %s", completion_object_path);

  ggml_language_model_completion_proxy_new (
    state->dbus_connection,
    G_DBUS_PROXY_FLAGS_NONE,
    NULL,
    completion_object_path,
    NULL,
    on_completion_object_ready,
    g_steal_pointer (&state)
  );
}

static void
on_language_model_service_session_proxy_ready (GObject      *source_object,
                                               GAsyncResult *result,
                                               gpointer      user_data)
{
  g_autoptr(GGMLLanguageModelClientState) state = user_data;
  g_autoptr(GError) error = NULL;
  g_autoptr(GGMLLanguageModelServiceSession) session_proxy = ggml_language_model_service_session_proxy_new_finish (result, &error);

  if (session_proxy == NULL)
    {
      g_error ("Failed to create LanguageModelSessionProxy object: %s", error->message);
      g_main_loop_quit (state->loop);
      return;
    }

  state->session_proxy = g_object_ref (session_proxy);

  GVariantBuilder builder;
  g_variant_builder_init (&builder, G_VARIANT_TYPE_ARRAY);
  g_variant_builder_add (&builder, "{sv}", "n_params", g_variant_new_string ("117M"));
  g_variant_builder_add (&builder, "{sv}", "quantization", g_variant_new_string ("f16"));
  g_autoptr(GVariant) properties = g_variant_ref_sink (g_variant_builder_end (&builder));

  /* Now lets create a cursor and start doing some inference */
  ggml_language_model_service_session_call_create_completion (state->session_proxy,
                                                              "gpt2",
                                                              properties,
                                                              "The meaning of life is:",
                                                              128,
                                                              G_DBUS_CALL_FLAGS_NONE,
                                                              -1,
                                                              NULL,
                                                              on_call_create_completion_reply,
                                                              g_steal_pointer (&state));

  g_message ("Created session proxy");
}

static void
on_created_private_connection (GObject      *source_object,
                               GAsyncResult *result,
                               gpointer      user_data)
{
  g_autoptr(GGMLLanguageModelClientState) state = user_data;
  g_autoptr(GError) error = NULL;
  g_autoptr(GDBusConnection) dbus_connection = g_dbus_connection_new_finish (result,
                                                                             &error);

  if (dbus_connection == NULL)
    {
      g_error ("Failed create client side of private connection: %s", error->message);
      g_main_loop_quit (state->loop);
      return;
    }

  state->dbus_connection = g_object_ref (dbus_connection);

  g_message ("Created private connection");

  ggml_language_model_service_session_proxy_new (
    dbus_connection,
    G_DBUS_PROXY_FLAGS_NONE,
    NULL, /* Not a bus connection */
    "/org/ggml_gobject/LanguageModelSession",
    NULL,
    on_language_model_service_session_proxy_ready,
    g_steal_pointer (&state)
 );
}

static void
on_call_open_session_reply (GObject      *source_object,
                            GAsyncResult *result,
                            gpointer      user_data)
{
  GGMLLanguageModelService *service_proxy = GGML_LANGUAGE_MODEL_SERVICE (source_object);
  g_autoptr(GGMLLanguageModelClientState) state = user_data;
  g_autoptr(GError) error = NULL;
  g_autoptr(GUnixFDList) fd_list = NULL;

  if (!ggml_language_model_service_call_open_session_finish (service_proxy,
                                                             &fd_list,
                                                             result,
                                                             &error))
    {
      g_error ("Failed to call OpenSession: %s", error->message);
      g_main_loop_quit (state->loop);
      return;
    }

  g_message ("Called OpenSession. Got %i fds", g_unix_fd_list_get_length (fd_list));

  int n_fds = 0;
  g_autofree int *fds = g_unix_fd_list_steal_fds (fd_list, &n_fds);

  g_autoptr(GInputStream) input_stream = g_unix_input_stream_new (fds[0], TRUE);
  g_autoptr(GOutputStream) output_stream = g_unix_output_stream_new (fds[1], TRUE);
  g_autoptr(GIOStream) io_stream = simple_io_stream_new (input_stream, output_stream);

  g_dbus_connection_new (io_stream,
                         NULL,
                         G_DBUS_CONNECTION_FLAGS_AUTHENTICATION_CLIENT,
                         NULL,
                         NULL,
                         on_created_private_connection,
                         g_steal_pointer (&state));
}

static void
on_language_model_service_proxy_ready (GObject      *source_object,
                                       GAsyncResult *result,
                                       gpointer      user_data)
{
  g_autoptr(GGMLLanguageModelClientState) state = user_data;
  g_autoptr(GError) error = NULL;
  g_autoptr(GGMLLanguageModelService) proxy = ggml_language_model_service_proxy_new_for_bus_finish (result, &error);

  if (proxy == NULL)
    {
      g_error ("Failed to create proxy: %s", error->message);
      g_main_loop_quit (state->loop);
      return;
    }

  state->service_proxy = g_object_ref (proxy);

  ggml_language_model_service_call_open_session (state->service_proxy,
                                                 G_DBUS_CALL_FLAGS_NONE,
                                                 -1,
                                                 NULL,
                                                 NULL,
                                                 on_call_open_session_reply,
                                                 g_steal_pointer (&state));

  g_message ("Created proxy");
}

static gboolean
on_main_loop_started (gpointer data)
{
  g_autoptr(GMainLoop) loop = data;
  g_autoptr(GGMLLanguageModelClientState) state = ggml_language_model_client_state_new (loop);

  g_message ("Started loop");

  ggml_language_model_service_proxy_new_for_bus (
    G_BUS_TYPE_SESSION,
    G_DBUS_PROXY_FLAGS_NONE,
    "org.ggml_gobject.LanguageModelService",
    "/org/ggml_gobject/LanguageModelService",
    NULL,
    on_language_model_service_proxy_ready,
    g_steal_pointer (&state)
 );

  return G_SOURCE_REMOVE;
}

int
main (int argc, char **argv)
{
  g_autoptr(GMainLoop) loop = g_main_loop_new (NULL, TRUE);

  g_idle_add (on_main_loop_started, g_main_loop_ref (loop));
  g_main_loop_run (loop);

  return 0;
}
