/*
 * examples/service-client/ggml-language-model-client.c
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
#include <gio/gio.h>
#include <ggml-gobject/dbus/ggml-client-session.h>
#include <ggml-gobject/dbus/ggml-client-language-model-cursor.h>

typedef struct {
  GMainLoop                     *loop;
  GGMLClientSession             *client_session;
  GGMLClientLanguageModelCursor *cursor;
  size_t                         ref_count;
} GGMLLanguageModelClientState;

static GGMLLanguageModelClientState *
ggml_language_model_client_state_new (GMainLoop *loop)
{
  GGMLLanguageModelClientState *state = g_new0 (GGMLLanguageModelClientState, 1);
  state->loop = g_main_loop_ref (loop);
  state->client_session = NULL;
  state->cursor = NULL;
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
      g_clear_pointer (&state->cursor, ggml_client_language_model_cursor_unref);
      g_clear_pointer (&state->client_session, ggml_client_session_unref);
      g_clear_pointer (&state->loop, g_main_loop_unref);
      g_clear_pointer (&state, g_free);
    }
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLLanguageModelClientState, ggml_language_model_client_state_unref)

static void
on_new_chunk_from_completion (const char *chunk,
                              gpointer    user_data)
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

  completed_string = ggml_client_language_model_cursor_exec_stream_finish (state->cursor,
                                                                           result,
                                                                           NULL,
                                                                           &error);

  if (completed_string == NULL)
    {
      g_error ("Error when calling LanguageModelCursor.exec(): %s", error->message);
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
  g_autoptr(GGMLLanguageModelClientState) state = user_data;
  g_autoptr(GError) error = NULL;
  g_autoptr(GGMLClientLanguageModelCursor) cursor = ggml_client_session_start_completion_finish (result, &error);

  if (cursor == NULL)
    {
      g_error ("Failed to create LanguageModelCursor object: %s", error->message);
      g_main_loop_quit (state->loop);
      return;
    }

  state->cursor = ggml_client_language_model_cursor_ref (cursor);

  ggml_client_language_model_cursor_exec_stream_async (cursor,
                                                       64,
                                                       2,
                                                       NULL,
                                                       on_new_chunk_from_completion,
                                                       NULL,
                                                       NULL,
                                                       on_done_complete_exec,
                                                       g_steal_pointer (&state));
}

static void
on_client_session_ready (GObject      *source_object,
                         GAsyncResult *result,
                         gpointer      user_data)
{
  g_autoptr(GGMLLanguageModelClientState) state = user_data;
  g_autoptr(GError) error = NULL;
  g_autoptr(GGMLClientSession) client_session = ggml_client_session_new_finish (result, &error);

  if (client_session == NULL)
    {
      g_error ("Failed to create ClientSession object: %s", error->message);
      g_main_loop_quit (state->loop);
      return;
    }

  state->client_session = ggml_client_session_ref (client_session);

  ggml_client_session_start_completion_async (state->client_session,
                                              "gpt2",
                                              "117M",
                                              "f16",
                                              "The meaning of life is:",
                                              128,
                                              NULL,
                                              on_completion_object_ready,
                                              g_steal_pointer (&state));

  g_message ("Created session proxy");
}

static gboolean
on_main_loop_started (gpointer data)
{
  g_autoptr(GMainLoop) loop = data;
  g_autoptr(GGMLLanguageModelClientState) state = ggml_language_model_client_state_new (loop);

  g_message ("Started loop");

  ggml_client_session_new_async (NULL,
                                 on_client_session_ready,
                                 g_steal_pointer (&state));

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
