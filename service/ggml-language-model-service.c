/*
 * service/ggml-language-model-service.c
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

#include <glib-object.h>
#include <glib-unix.h>
#include <gio/gio.h>
#include <gio/gunixinputstream.h>
#include <gio/gunixoutputstream.h>
#include <gio/gunixfdlist.h>
#include <ggml-gobject/dbus/ggml-service-dbus.h>
#include <ggml-gobject/ggml-gobject.h>

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

typedef struct _GGMLServiceState {
  GMainLoop                *loop;
  GGMLService *service_skeleton;
  GHashTable               *connections;
  GHashTable               *models;
  size_t                    ref_count;
} GGMLServiceState;

/* Forward declaration */
typedef struct _GGMLServiceConnection GGMLServiceConnection;
typedef struct _GGMLLanguageModelRef GGMLLanguageModelRef;
static void ggml_service_connection_unref (GGMLServiceConnection *conn);
static void ggml_language_model_ref_free (GGMLLanguageModelRef *ref);
gboolean ggml_service_handle_open_session (GGMLService           *object,
                                           GDBusMethodInvocation *invocation,
                                           GUnixFDList           *in_fd_list,
                                           gpointer               user_data);

static GGMLServiceState *
ggml_service_state_new (GMainLoop *loop)
{
  GGMLServiceState *state = g_new0 (GGMLServiceState, 1);
  state->loop = g_main_loop_ref (loop);
  state->service_skeleton = ggml_service_skeleton_new ();
  state->connections = g_hash_table_new_full (g_int_hash, NULL, (GDestroyNotify) ggml_service_connection_unref, NULL);
  state->models = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, (GDestroyNotify) ggml_language_model_ref_free);
  state->ref_count = 1;

  g_signal_connect (state->service_skeleton,
                    "handle-open-session",
                    G_CALLBACK (ggml_service_handle_open_session),
                    state);

  return state;
}

static GGMLServiceState *
ggml_service_state_ref (GGMLServiceState *state)
{
  ++state->ref_count;
  return state;
}

static void
ggml_service_state_unref (GGMLServiceState *state)
{
  if (--state->ref_count == 0)
    {
      g_clear_pointer (&state->connections, g_hash_table_unref);
      g_clear_pointer (&state->loop, g_main_loop_unref);
      g_clear_object (&state->service_skeleton);
      g_clear_pointer (&state, g_free);
    }
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLServiceState, ggml_service_state_unref)

typedef struct _GGMLServiceConnection {
  /* weak reference */
  GGMLServiceState   *parent_state;

  GIOStream          *stream;
  GDBusConnection    *dbus_connection;
  GGMLSession        *session;
  GHashTable         *cursors;
  size_t              cursor_serial;
  size_t              ref_count;
} GGMLServiceConnection;


typedef struct _GGMLSessionCompletion GGMLServiceConnectionCompletion;
void ggml_session_completion_unref (GGMLServiceConnectionCompletion *);

GGMLServiceConnection *
ggml_service_connection_new (GGMLServiceState *parent_state,
                             GIOStream        *stream)
{
  GGMLServiceConnection *connection = g_new0 (GGMLServiceConnection, 1);

  connection->parent_state = parent_state;
  connection->stream = g_object_ref (stream);
  connection->session = ggml_session_skeleton_new ();
  connection->cursors = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, (GDestroyNotify) ggml_session_completion_unref);
  connection->ref_count = 1;

  return connection;
}

GGMLServiceConnection *
ggml_service_connection_ref (GGMLServiceConnection *connection)
{
  ++connection->ref_count;
  return connection;
}

void
ggml_service_connection_unref (GGMLServiceConnection *connection)
{
  if (--connection->ref_count == 0)
    {
      g_clear_pointer (&connection->cursors, g_hash_table_unref);
      g_clear_object (&connection->session);
      g_clear_object (&connection->dbus_connection);
      g_clear_object (&connection->stream);
      g_clear_pointer (&connection, g_free);
    }
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLServiceConnection, ggml_service_connection_unref)

static char *
language_model_to_key (const char *model_name,
                       GVariant   *properties)
{
  g_autofree char *variant_props_string = g_variant_print (properties, FALSE);
  const char *strings[] = {
    model_name,
    variant_props_string,
    NULL
  };

  return g_strjoinv ("-", (char **) strings);
}

typedef struct _GGMLLanguageModelRef {
  GGMLLanguageModel *model;
  size_t             use_count;
} GGMLLanguageModelRef;

GGMLLanguageModelRef *
ggml_language_model_ref_new (GGMLLanguageModel *model)
{
  GGMLLanguageModelRef *ref = g_new0 (GGMLLanguageModelRef, 1);

  ref->model = ggml_language_model_ref (model);
  ref->use_count = 0;

  return ref;
}

void
ggml_language_model_ref_free (GGMLLanguageModelRef *ref)
{
  g_assert (ref->use_count == 0);

  g_clear_pointer (&ref->model, ggml_language_model_unref);
  g_clear_pointer (&ref, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLLanguageModelRef, ggml_language_model_ref_free);

typedef void (*ExecWithModelCallback) (GGMLLanguageModelRef *model,
                                       GError               *error,
                                       gpointer              user_data);

static gboolean
get_defined_model (const char                *model,
                   const char                *num_params,
                   GGMLDefinedLanguageModel  *out_defined_language_model,
                   GError                   **error)
{
  g_assert (out_defined_language_model != NULL);

  if (g_strcmp0 (model, "gpt2") == 0)
    {
      if (num_params == NULL)
        {
          *out_defined_language_model = GGML_DEFINED_LANGUAGE_MODEL_GPT2P117M;
        }

      if (g_strcmp0 (num_params, "117M") == 0)
        {
          *out_defined_language_model = GGML_DEFINED_LANGUAGE_MODEL_GPT2P117M;
          return TRUE;
        }
      if (g_strcmp0 (num_params, "345M") == 0)
        {
          *out_defined_language_model = GGML_DEFINED_LANGUAGE_MODEL_GPT2P345M;
          return TRUE;
        }
      if (g_strcmp0 (num_params, "774M") == 0)
        {
          *out_defined_language_model = GGML_DEFINED_LANGUAGE_MODEL_GPT2P774M;
          return TRUE;
        }
      if (g_strcmp0 (num_params, "1558M") == 0)
        {
          *out_defined_language_model = GGML_DEFINED_LANGUAGE_MODEL_GPT2P1558M;
          return TRUE;
        }
    }

  g_set_error (error,
               G_IO_ERROR,
               G_IO_ERROR_FAILED,
               "Could not find model matching (name: %s, params: %s)",
               model,
               num_params);
  return FALSE;
}

static gboolean
get_quantization_type (const char    *model_quantization,
                       GGMLDataType  *out_quantize_type,
                       GError       **error)
{
  if (model_quantization == NULL)
    {
      *out_quantize_type = GGML_DATA_TYPE_F16;
      return TRUE;
    }

  if (g_strcmp0 (model_quantization, "f16") == 0)
    {
      *out_quantize_type = GGML_DATA_TYPE_F16;
      return TRUE;
    }

  if (g_strcmp0 (model_quantization, "q8_0") == 0)
    {
      *out_quantize_type = GGML_DATA_TYPE_Q8_0;
      return TRUE;
    }

  if (g_strcmp0 (model_quantization, "q5_0") == 0)
    {
      *out_quantize_type = GGML_DATA_TYPE_Q5_0;
      return TRUE;
    }

  if (g_strcmp0 (model_quantization, "q5_1") == 0)
    {
      *out_quantize_type = GGML_DATA_TYPE_Q5_1;
      return TRUE;
    }

  if (g_strcmp0 (model_quantization, "q4_0") == 0)
    {
      *out_quantize_type = GGML_DATA_TYPE_Q4_0;
      return TRUE;
    }

  if (g_strcmp0 (model_quantization, "q4_1") == 0)
    {
      *out_quantize_type = GGML_DATA_TYPE_Q4_1;
      return TRUE;
    }

  g_set_error (error,
               G_IO_ERROR,
               G_IO_ERROR_FAILED,
               "No such quantization type: %s (valid options: f16, q8_0, q5_0, q5_1, q4_0, q4_1)",
               model_quantization);
  return FALSE;
}

typedef struct {
  GGMLServiceState      *service_state;
  char                  *key;
  ExecWithModelCallback  callback;
  gpointer               user_data;
} ExecWithModelClosure;

static ExecWithModelClosure *
exec_with_model_closure_new (GGMLServiceState      *service_state,
                             char                  *model_key,
                             ExecWithModelCallback  callback,
                             gpointer               user_data)
{
  ExecWithModelClosure *closure = g_new0 (ExecWithModelClosure, 1);

  closure->service_state = ggml_service_state_ref (service_state);
  closure->key = g_strdup (model_key);
  closure->callback = callback;
  closure->user_data = user_data;

  return closure;
}

void
exec_with_model_closure_free (ExecWithModelClosure *closure)
{
  g_clear_pointer (&closure->service_state, ggml_service_state_unref);
  g_clear_pointer (&closure->key, g_free);
  g_clear_pointer (&closure, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (ExecWithModelClosure, exec_with_model_closure_free);

void
on_loaded_defined_model (GObject      *source_object,
                         GAsyncResult *result,
                         gpointer      user_data)
{
  g_autoptr(GError) error = NULL;
  g_autoptr(GGMLLanguageModel) language_model = ggml_language_model_load_defined_from_istream_finish (result, &error);
  g_autoptr(ExecWithModelClosure) closure = user_data;

  if (language_model == NULL)
    {
      (*closure->callback) (NULL, error, closure->user_data);
      return;
    }

  /* Call the closure callback with the ref */
  g_autoptr(GGMLLanguageModelRef) language_model_ref = ggml_language_model_ref_new (language_model);
  (*closure->callback) (language_model_ref, NULL, closure->user_data);

  /* If we created the language model, we should now add it to our dictionary
   * so that we can keep track of it later. */
  g_hash_table_insert (closure->service_state->models,
                       closure->key,
                       g_steal_pointer (&language_model_ref));
}

void
ggml_service_ref_model_async (GGMLServiceState      *service,
                              const char            *model,
                              GVariant              *properties,
                              ExecWithModelCallback  callback,
                              gpointer               user_data)
{
  /* Check to see if we have the model already.
   *
   * XXX: Right now this doesn't handle concurrent requests. */
  g_autoptr(GError) error = NULL;
  g_autofree char *model_key = language_model_to_key (model, properties);
  GGMLLanguageModelRef *language_model_ref = g_hash_table_lookup (service->models,
                                                                  model_key);

  if (language_model_ref != NULL)
    {
      (*callback) (language_model_ref, NULL, user_data);
      return;
    }

  /* We have to load the model first */
  g_autofree char *model_num_params = NULL;
  g_autofree char *model_quantization = NULL;

  GVariantIter iter;
  char *key;
  GVariant *value;
  g_variant_iter_init (&iter, properties);

  while (g_variant_iter_loop (&iter, "{sv}", &key, &value))
    {
      if (g_strcmp0 (key, "n_params") == 0)
        {
          model_num_params = g_variant_dup_string (value, NULL);
        }

      if (g_strcmp0 (key, "quantization") == 0)
        {
          model_quantization = g_variant_dup_string (value, NULL);
        }
    }

  GGMLDefinedLanguageModel defined_model;
  GGMLDataType quantize_type;
  
  if (!get_defined_model (model, model_num_params, &defined_model, &error))
    {
      (*callback) (NULL, error, user_data);
      return;
    }

  if (!get_quantization_type (model_quantization, &quantize_type, &error))
    {
      (*callback) (NULL, error, user_data);
      return;
    }

  /* Now we create the model, asynchronously */
  g_autoptr(GGMLModelConfig) config = ggml_model_config_new ();
  ggml_model_config_set_quantization_config (config,
                                             quantize_type,
                                             ggml_gpt_model_quantization_regexes (),
                                             NULL);

  g_autoptr(GGMLCachedModelIstream) istream = ggml_language_model_stream_from_cache (defined_model, &error);

  if (istream == NULL)
    {
      (*callback) (NULL, error, user_data);
      return;
    }

  g_autoptr(ExecWithModelClosure) closure = exec_with_model_closure_new (service,
                                                                         model_key,
                                                                         callback,
                                                                         user_data);
  ggml_language_model_load_defined_from_istream_async (defined_model,
                                                       G_INPUT_STREAM (istream),
                                                       config,
                                                       NULL,
                                                       on_loaded_defined_model,
                                                       g_steal_pointer (&closure));

}

typedef struct {
  GGMLSession           *object;
  GDBusMethodInvocation *invocation;
  GGMLServiceConnection *conn;
  char                  *prompt;
  int                    max_tokens;
} CreateCompletionClosure;

CreateCompletionClosure *
create_completion_closure_new (GGMLSession           *object,
                               GDBusMethodInvocation *invocation,
                               GGMLServiceConnection *conn,
                               const char            *prompt,
                               int                    max_tokens)
{
  CreateCompletionClosure *closure = g_new0 (CreateCompletionClosure, 1);

  closure->object = g_object_ref (object);
  closure->invocation = g_object_ref (invocation);
  closure->conn = ggml_service_connection_ref (conn);
  closure->prompt = g_strdup (prompt);
  closure->max_tokens = max_tokens;

  return closure;
}

void
create_completion_closure_free (CreateCompletionClosure *closure)
{
  g_clear_object (&closure->object);
  g_clear_object (&closure->invocation);
  g_clear_pointer (&closure->conn, ggml_service_connection_unref);
  g_clear_pointer (&closure->prompt, g_free);
  g_clear_pointer (&closure, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (CreateCompletionClosure, create_completion_closure_free);

typedef struct _GGMLSessionCompletion {
  /* weak reference */
  GGMLSession   *parent_session;

  GGMLLanguageModelCompletionCursor *cursor;
  GGMLLanguageModelCompletion       *completion_skeleton;
  size_t ref_count;
} GGMLSessionCompletion;

GGMLSessionCompletion *
ggml_session_completion_new (GGMLSession *parent_session,
                             GGMLLanguageModel  *model,
                             const gchar        *prompt,
                             gint                max_tokens)
{
  GGMLSessionCompletion *completion = g_new0 (GGMLSessionCompletion, 1);

  completion->parent_session = parent_session;
  completion->completion_skeleton = ggml_language_model_completion_skeleton_new ();
  completion->cursor = ggml_language_model_create_completion (model, prompt, max_tokens);
  completion->ref_count = 1;

  return completion;
}

void
ggml_session_completion_unref (GGMLSessionCompletion *completion)
{
  if (--completion->ref_count == 0)
    {
      g_clear_object  (&completion->completion_skeleton);
      g_clear_pointer (&completion->cursor, ggml_language_model_completion_cursor_unref);
      g_clear_pointer (&completion, g_free);
    }
}

GGMLSessionCompletion *
ggml_session_completion_ref (GGMLSessionCompletion *completion)
{
  ++completion->ref_count;
  return completion;
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLSessionCompletion, ggml_session_completion_unref);

typedef struct {
  GGMLLanguageModelCompletion *completion_skeleton;
  GDBusMethodInvocation       *invocation;
  GPtrArray                   *stream_chunks;
  GGMLSessionCompletion       *completion;
} HandleCompletionExecClosure;

HandleCompletionExecClosure *
handle_completion_exec_closure_new (GGMLLanguageModelCompletion *completion_skeleton,
                                    GDBusMethodInvocation       *invocation,
                                    GGMLSessionCompletion       *completion,
                                    size_t                       num_tokens)
{
  HandleCompletionExecClosure *closure = g_new0 (HandleCompletionExecClosure, 1);

  closure->completion_skeleton = g_object_ref (completion_skeleton);
  closure->invocation = g_object_ref (invocation);
  closure->stream_chunks = g_ptr_array_new_full (num_tokens, g_free);
  closure->completion = ggml_session_completion_ref (completion);

  return closure;
}

void
handle_completion_exec_closure_free (HandleCompletionExecClosure *closure)
{
  g_clear_object (&closure->completion_skeleton);
  g_clear_object (&closure->invocation);
  g_clear_pointer (&closure->stream_chunks, g_ptr_array_unref);
  g_clear_pointer (&closure->completion, ggml_session_completion_unref);
  g_clear_pointer (&closure, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (HandleCompletionExecClosure, handle_completion_exec_closure_free)

void
on_new_tokens_for_cursor (const char *decoded,
                          gboolean    is_complete_eos,
                          gpointer    user_data)
{
  HandleCompletionExecClosure *closure = user_data;

  /* We emit the "new-tokens" signal from the skeleton */
  ggml_language_model_completion_emit_new_chunk (closure->completion_skeleton,
                                                 decoded);

  /* We also store the chunk in our closure */
  g_ptr_array_add (closure->stream_chunks, g_strdup (decoded));
}

void
on_done_exec_stream_for_cursor (GObject      *source_object,
                                GAsyncResult *result,
                                gpointer      user_data)
{
  g_autoptr(GError) error = NULL;
  g_autoptr(HandleCompletionExecClosure) closure = user_data;

  if (!ggml_language_model_completion_cursor_exec_stream_finish (closure->completion->cursor,
                                                                 result,
                                                                 &error))
    {
      g_dbus_method_invocation_return_gerror (closure->invocation, error);
      return;
    }

  g_message ("Done with streaming");

  g_ptr_array_add (closure->stream_chunks, NULL);

  g_autofree char *completion = g_strjoinv ("", (char **) closure->stream_chunks->pdata);

  ggml_language_model_completion_complete_exec (closure->completion_skeleton,
                                                closure->invocation,
                                                completion);
}

gboolean
on_handle_completion_exec (GGMLLanguageModelCompletion *completion_skeleton,
                           GDBusMethodInvocation       *invocation,
                           int                          num_tokens,
                           gpointer                     user_data)
{
  g_autoptr(GGMLSessionCompletion) completion = user_data;
  HandleCompletionExecClosure *closure = handle_completion_exec_closure_new (completion_skeleton,
                                                                             invocation,
                                                                             completion,
                                                                             num_tokens);

  ggml_language_model_completion_cursor_exec_stream_async (completion->cursor,
                                                           num_tokens,
                                                           2,
                                                           NULL,
                                                           on_new_tokens_for_cursor,
                                                           closure,
                                                           NULL,
                                                           on_done_exec_stream_for_cursor,
                                                           closure);

  return TRUE;
}

void
on_create_completion_obtained_model_ref (GGMLLanguageModelRef *ref,
                                         GError               *error,
                                         gpointer              user_data)
{
  g_autoptr(CreateCompletionClosure) closure = user_data;

  if (error != NULL)
    {
      g_dbus_method_invocation_return_gerror (closure->invocation,
                                              error);
    }

  /* Create a completion */
  g_autoptr(GGMLSessionCompletion) completion = ggml_session_completion_new (closure->conn->session,
                                                                             ref->model,
                                                                             closure->prompt,
                                                                             closure->max_tokens);

  g_autofree char *completion_object_path = g_strdup_printf ("/org/ggml/LanguageModelCompletion/%zu",
                                                            closure->conn->cursor_serial++);

  /* Expose the /org/ggml/LanguageModelCompletion/n object */
  if (!g_dbus_interface_skeleton_export (G_DBUS_INTERFACE_SKELETON (completion->completion_skeleton),
                                         closure->conn->dbus_connection,
                                         completion_object_path,
                                         &error))
    {
      g_error ("Failed to export LanguageModelCompletion object: %s", error->message);
      g_main_loop_quit (closure->conn->parent_state->loop);
      return;
    }

  /* Now that we have that, we can add the connnection to the hash table */
  g_hash_table_insert (closure->conn->cursors,
                       g_strdup (completion_object_path),
                       ggml_session_completion_ref (completion));

  /* Now we take the ref and create a cursor on the bus */
  ++ref->use_count;

  ggml_session_complete_create_completion (closure->object,
                                           closure->invocation,
                                           completion_object_path);

  /* Handle the exec() method call */
  g_signal_connect (completion->completion_skeleton,
                    "handle-exec",
                    G_CALLBACK (on_handle_completion_exec),
                    completion);

  g_message ("Created cursor, exposed object at path %s\n", completion_object_path);
}

gboolean
on_handle_create_completion (GGMLSession           *object,
                             GDBusMethodInvocation *invocation,
                             const char            *model,
                             GVariant              *properties,
                             const char            *prompt,
                             int                    max_tokens,
                             gpointer               user_data)
{
  GGMLServiceConnection *conn = user_data;
  g_autoptr(CreateCompletionClosure) closure = create_completion_closure_new (object,
                                                                              invocation,
                                                                              conn,
                                                                              prompt,
                                                                              max_tokens);

  ggml_service_ref_model_async (conn->parent_state,
                                model,
                                properties,
                                on_create_completion_obtained_model_ref,
                                g_steal_pointer (&closure));
  return TRUE;
}

void
on_private_server_bus_connection_closed (GDBusConnection *dbus_connection,
                                         gboolean         remote_peer_vanished,
                                         GError          *error,
                                         gpointer         user_data)
{
  GGMLServiceConnection *conn = user_data;
  GHashTable *connections = conn->parent_state->connections;

  /* Remove the connection from the set of connections */
  g_hash_table_remove (connections, conn);
  g_message ("Removed connection %p, connections table has %i objects", conn, g_hash_table_size (connections));
}


void
on_created_private_bus_server_connection (GObject      *object,
                                          GAsyncResult *result,
                                          gpointer      user_data)
{
  g_autoptr(GError) error = NULL;
  GGMLServiceConnection *conn = user_data;
  g_autoptr(GDBusConnection) dbus_connection = g_dbus_connection_new_finish (result, &error);

  if (dbus_connection == NULL)
    {
      g_error ("Error creating connection: %s", error->message);
      return;
    }

  conn->dbus_connection = g_object_ref (dbus_connection);

  g_signal_connect (conn->dbus_connection,
                    "closed",
                    G_CALLBACK (on_private_server_bus_connection_closed),
                    conn);

  g_message ("Created private connection");

  /* Expose the /org/ggml/Session object */
  if (!g_dbus_interface_skeleton_export (G_DBUS_INTERFACE_SKELETON (conn->session),
                                         dbus_connection,
                                         "/org/ggml/Session",
                                         &error))
    {
      g_error ("Failed to export Session object: %s", error->message);
      g_main_loop_quit (conn->parent_state->loop);
      return;
    }

  g_signal_connect (conn->session,
                    "handle-create-completion",
                    G_CALLBACK (on_handle_create_completion),
                    conn);

  g_message ("Exported session");
}

gboolean
ggml_service_handle_open_session (GGMLService           *object,
                                  GDBusMethodInvocation *invocation,
                                  GUnixFDList           *in_fd_list,
                                  gpointer               user_data)
{
  GGMLServiceState *state = user_data;
  g_autoptr(GError) error = NULL;
  g_autoptr(GUnixFDList) fd_list = g_unix_fd_list_new ();
  int client_to_server_fds[2];
  int server_to_client_fds[2];

  if (!g_unix_open_pipe (client_to_server_fds, FD_CLOEXEC, &error))
    {
      g_dbus_method_invocation_return_gerror (invocation, error);
      return TRUE;
    }

  if (!g_unix_open_pipe (server_to_client_fds, FD_CLOEXEC, &error))
    {
      g_dbus_method_invocation_return_gerror (invocation, error);
      return TRUE;
    }

  if (g_unix_fd_list_append (fd_list, server_to_client_fds[0], &error) == -1)
    {
      g_dbus_method_invocation_return_gerror (invocation, error);
      return TRUE;
    }

  if (g_unix_fd_list_append (fd_list, client_to_server_fds[1], &error) == -1)
    {
      g_dbus_method_invocation_return_gerror (invocation, error);
      return TRUE;
    }

  close (server_to_client_fds[0]);
  close (client_to_server_fds[1]);

  /* Complete opening the session - this will cause the client to try
   * and connect on the other end of the pipes */
  ggml_service_complete_open_session (object,
                                      invocation,
                                      fd_list);

  g_autoptr(GInputStream) input_stream = g_unix_input_stream_new (client_to_server_fds[0], TRUE);
  g_autoptr(GOutputStream) output_stream = g_unix_output_stream_new (server_to_client_fds[1], TRUE);
  g_autoptr(GIOStream) io_stream = simple_io_stream_new (input_stream,
                                                         output_stream);

  g_autoptr(GGMLServiceConnection) conn = ggml_service_connection_new (state, io_stream);
  g_hash_table_insert (state->connections, ggml_service_connection_ref (conn), NULL);

  g_autofree char *guid = g_dbus_generate_guid ();
  g_dbus_connection_new (conn->stream,
                         guid,
                         G_DBUS_CONNECTION_FLAGS_AUTHENTICATION_SERVER |
                         G_DBUS_CONNECTION_FLAGS_AUTHENTICATION_ALLOW_ANONYMOUS,
                         NULL,
                         NULL,
                         on_created_private_bus_server_connection,
                         conn);

  return TRUE;
}

static void
on_acquired_bus (GDBusConnection *connection,
                 const char      *name,
                 gpointer         user_data)
{
  g_message ("Acquired bus");
}

static void
on_acquired_name (GDBusConnection *connection,
                  const char      *name,
                  gpointer         user_data)
{
  GGMLServiceState *state = user_data;
  g_autoptr(GError) error = NULL;

  g_message ("Acquired name");

  if (!g_dbus_interface_skeleton_export (G_DBUS_INTERFACE_SKELETON (state->service_skeleton),
                                         connection,
                                         "/org/ggml/Service",
                                         &error))
    {
      g_error ("Failed to export Service object: %s", error->message);
      g_main_loop_quit (state->loop);
      return;
    }
}

static void
on_lost_name (GDBusConnection *connection,
              const char      *name,
              gpointer         user_data)
{
  GGMLServiceState *state = user_data;
  g_message ("Lost name");

  g_main_loop_quit (state->loop);
}

static gboolean
on_main_loop_started (gpointer data)
{
  GGMLServiceState *state = data;

  g_message ("Started loop");

  g_bus_own_name (G_BUS_TYPE_SESSION,
                  "org.ggml.Service",
                  G_BUS_NAME_OWNER_FLAGS_REPLACE |
                  G_BUS_NAME_OWNER_FLAGS_ALLOW_REPLACEMENT,
                  on_acquired_bus,
                  on_acquired_name,
                  on_lost_name,
                  state,
                  NULL);

  return G_SOURCE_REMOVE;
}

int main (int argc, char **argv)
{
  g_autoptr(GMainLoop) loop = g_main_loop_new (NULL, TRUE);
  g_autoptr(GGMLServiceState) state = ggml_service_state_new (loop);

  g_idle_add (on_main_loop_started, state);
  g_main_loop_run (loop);
  return 0;
}
