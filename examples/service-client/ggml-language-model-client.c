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

#define GETTEXT_PACKAGE "ggml-gobject"
#include <glib/gi18n-lib.h>

typedef struct {
  GMainLoop                     *loop;

  char                          *prompt;
  char                          *model;
  char                          *variant;
  char                          *quantization;
  int                            max_size;
  int                            top_k;
  float                          top_p;
  int                            seed;

  GGMLClientSession             *client_session;
  GGMLClientLanguageModelCursor *cursor;
  size_t                         ref_count;
} GGMLLanguageModelClientState;

static GGMLLanguageModelClientState *
ggml_language_model_client_state_new (GMainLoop *loop,
                                      char      *prompt,
                                      char      *model,
                                      char      *variant,
                                      char      *quantization,
                                      int        max_size,
                                      int        top_k,
                                      float      top_p,
                                      int        seed)
{
  GGMLLanguageModelClientState *state = g_new0 (GGMLLanguageModelClientState, 1);
  state->loop = g_main_loop_ref (loop);
  state->prompt = g_strdup (prompt);
  state->model = g_strdup (model);
  state->variant = g_strdup (variant);
  state->quantization = g_strdup (quantization);
  state->max_size = max_size;
  state->top_k = top_k;
  state->top_p = top_p;
  state->seed = seed;

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
      g_clear_pointer (&state->prompt, g_free);
      g_clear_pointer (&state->model, g_free);
      g_clear_pointer (&state->variant, g_free);
      g_clear_pointer (&state->quantization, g_free);
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
                                                       state->max_size,
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

  GVariantBuilder builder;
  g_variant_builder_init (&builder, G_VARIANT_TYPE ("a{sv}"));
  g_variant_builder_add (&builder, "{sv}", "top_k", g_variant_new_uint32 (state->top_k));
  g_variant_builder_add (&builder, "{sv}", "top_p", g_variant_new_double (state->top_p));

  if (state->seed != -1)
    {
      g_variant_builder_add (&builder, "{sv}", "sampler_seed", g_variant_new_uint32 (state->seed));
    }

  g_autoptr(GVariant) properties = g_variant_ref_sink (g_variant_builder_end (&builder));

  ggml_client_session_start_completion_async (state->client_session,
                                              state->model != NULL ? state->model : "gpt2",
                                              state->variant != NULL ? state->variant : "117M",
                                              state->quantization != NULL ? state->quantization : "f16",
                                              state->prompt != NULL ? state->prompt : "The meaning of life is:",
                                              state->max_size,
                                              properties,
                                              NULL,
                                              on_completion_object_ready,
                                              g_steal_pointer (&state));

  g_message ("Created session proxy");
}

static gboolean
on_main_loop_started (gpointer data)
{
  g_autoptr(GGMLLanguageModelClientState) state = data;

  g_message ("Started loop");

  ggml_client_session_new_async (NULL,
                                 on_client_session_ready,
                                 g_steal_pointer (&state));

  return G_SOURCE_REMOVE;
}

int
main (int argc, char **argv)
{
  g_autoptr(GError) error = NULL;
  g_autofree char *prompt = NULL;
  g_autofree char *model = NULL;
  g_autofree char *variant = NULL;
  g_autofree char *quantization = NULL;
  int seed = -1;

  int max_size = 128;
  int top_k = 500;
  double top_p = 0.6;

  GOptionEntry options[] = {
    { "prompt", 'p', 0, G_OPTION_ARG_STRING, &prompt, "Prompt to use", "P" },
    { "model", 'm', 0, G_OPTION_ARG_STRING, &model, "Model to use", "M" },
    { "model-variant", 'v', 0, G_OPTION_ARG_STRING, &variant, "Variant of model (eg, 117M)", "V" },
    { "quantization", 'q', 0, G_OPTION_ARG_STRING, &prompt, "Quantization to use (f32, q8_0, q4_0, q4_1, q5_0, q5_1", "Q" },
    { "max-size", 's', 0, G_OPTION_ARG_INT, &max_size, "Max number of tokens to generate", "S" },
    { "top-k", 'k', 0, G_OPTION_ARG_INT, &top_k, "Top-k tokens to consider", "K" },
    { "top-p", 'p', 0, G_OPTION_ARG_DOUBLE, &top_p, "Top-p probability mass to consider", "T" },
    { "seed", 'q', 0, G_OPTION_ARG_INT, &seed, "Seed to use for the random number generator", "Y" },
    { NULL }
  };
  g_autoptr(GOptionContext) context = g_option_context_new ("example LLM client");
  g_option_context_add_main_entries (context, options, GETTEXT_PACKAGE);
  if (!g_option_context_parse (context, &argc, &argv, &error))
    {
      g_error ("Could not parse options: %s", error->message);
      return 1;
    }

  g_autoptr(GMainLoop) loop = g_main_loop_new (NULL, TRUE);
  g_autoptr(GGMLLanguageModelClientState) state = ggml_language_model_client_state_new (
    loop,
    prompt,
    model,
    variant,
    quantization,
    max_size,
    top_k,
    top_p,
    seed
  );

  g_idle_add (on_main_loop_started, g_steal_pointer (&state));
  g_main_loop_run (loop);

  return 0;
}
