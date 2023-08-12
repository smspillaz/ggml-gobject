/*
 * ggml-gobject/dbus/ggml-client-language-model-cursor.c
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

#include <ggml-gobject/dbus/ggml-client-language-model-cursor.h>
#include <ggml-gobject/dbus/internal/ggml-client-language-model-cursor-internal.h>
#include <ggml-gobject/dbus/ggml-service-dbus.h>

struct _GGMLClientLanguageModelCursor {
  GGMLLanguageModelCompletion                *proxy;
  GGMLClientLanguageModelCursorChunkCallback  chunk_callback;
  gpointer                                    chunk_callback_data;
  GDestroyNotify                              chunk_callback_data_destroy;
  int32_t                                     new_chunk_connection;
  size_t                                      ref_count;
};

static void
on_new_chunk_from_completion (GGMLLanguageModelCompletion *completion,
                              const char                  *chunk,
                              gpointer                     user_data)
{
  GGMLClientLanguageModelCursor *cursor = user_data;

  if (cursor->chunk_callback != NULL)
    {
      (*cursor->chunk_callback) (chunk, cursor->chunk_callback_data);
    }
}

GGMLClientLanguageModelCursor *
ggml_client_language_model_cursor_new (GGMLLanguageModelCompletion                *proxy,
                                       GGMLClientLanguageModelCursorChunkCallback  chunk_callback,
                                       gpointer                                    chunk_callback_data,
                                       GDestroyNotify                              chunk_callback_data_destroy)
{
  GGMLClientLanguageModelCursor *cursor = g_new0 (GGMLClientLanguageModelCursor, 1);
  cursor->proxy = g_object_ref (proxy);
  cursor->chunk_callback = chunk_callback;
  cursor->chunk_callback_data = chunk_callback_data;
  cursor->chunk_callback_data_destroy = chunk_callback_data_destroy;
  cursor->ref_count = 1;

  /* Now connect to the new-chunk signal and also call
   * exec() on the proxy. We can interactively start printing the result
   * on the console */
  cursor->new_chunk_connection = g_signal_connect (cursor->proxy,
                                                   "new-chunk",
                                                   G_CALLBACK (on_new_chunk_from_completion),
                                                   cursor);

  return cursor;
}

/**
 * ggml_client_language_model_cursor_ref: (skip):
 * @cursor: A #GGMLClientLanguageModelCursor
 *
 * Increase ref count on this #GGMLClientLanguageModelCursor
 *
 * Returns: (transfer full): A #GGMLClientLanguageModelCursor
 */
GGMLClientLanguageModelCursor *
ggml_client_language_model_cursor_ref (GGMLClientLanguageModelCursor *cursor)
{
  ++cursor->ref_count;
  return cursor;
}

void
on_terminate_call_completed (GObject      *source_object,
                             GAsyncResult *result,
                             gpointer      user_data)
{
  g_autoptr(GError) error = NULL;

  if (!ggml_language_model_completion_call_terminate_finish (GGML_LANGUAGE_MODEL_COMPLETION (source_object),
                                                             result,
                                                             &error))
    {
      g_message ("Failed to destroy cursor on the remote end: %s", error->message);
      return;
    }

  g_message ("Successfuly destroyed cursor on the remote end");
}

/**
 * ggml_client_language_model_cursor_unref: (skip):
 * @cursor: A #GGMLClientLanguageModelCursor
 *
 * Decrease the ref count on this #GGMLClientLanguageModelCursor . When the ref count
 * drops to zero, then the cursor will be freed.
 *
 * Note this will also terminate the instance of the remote cursor on the
 * bus as well, so all the memory might not necessarily be freed at this point
 * but will be freed on a later main loop iteration.
 */
void
ggml_client_language_model_cursor_unref (GGMLClientLanguageModelCursor *cursor)
{
  if (--cursor->ref_count == 0)
    {
      if (cursor->new_chunk_connection != -1)
        {
          g_signal_handler_disconnect (cursor->proxy, cursor->new_chunk_connection);
          cursor->new_chunk_connection = -1;
        }

      /* If we still have the proxy, then we have to take another ref on the proxy at least
       * while we call terminate() on the remote side, this ensures that any memory on
       * the server side has been cleaned up. */
      if (cursor->proxy != NULL)
        {
          ggml_language_model_completion_call_terminate (cursor->proxy,
                                                         G_DBUS_CALL_FLAGS_NONE,
                                                         -1,
                                                         NULL,
                                                         on_terminate_call_completed,
                                                         NULL);
        }

      g_clear_pointer (&cursor->chunk_callback_data, cursor->chunk_callback_data_destroy);
      g_clear_object (&cursor->proxy);
      g_clear_pointer (&cursor, g_free);
    }
}

typedef struct {
  GGMLClientLanguageModelCursor *cursor;
  GAsyncReadyCallback            callback;
  gpointer                       user_data;
} CallCursorExecClosure;

CallCursorExecClosure *
call_cursor_exec_closure_new (GGMLClientLanguageModelCursor *cursor,
                              GAsyncReadyCallback            callback,
                              gpointer                       user_data)
{
  CallCursorExecClosure *closure = g_new0 (CallCursorExecClosure, 1);
  closure->cursor = ggml_client_language_model_cursor_ref (cursor);
  closure->callback = callback;
  closure->user_data = user_data;

  return closure;
}

void
call_cursor_exec_closure_free (CallCursorExecClosure *closure)
{
  g_clear_pointer (&closure->cursor, ggml_client_language_model_cursor_unref);
  g_clear_pointer (&closure, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (CallCursorExecClosure, call_cursor_exec_closure_free);

static void
on_done_complete_exec (GObject      *source_object,
                       GAsyncResult *result,
                       gpointer      user_data)
{
  g_autoptr(GError) error = NULL;
  g_autoptr(CallCursorExecClosure) closure = user_data;
  g_autoptr(GTask) task = g_task_new (NULL, NULL, closure->callback, closure->user_data);
  g_autofree char *completed_string = NULL;

  if (!ggml_language_model_completion_call_exec_finish (GGML_LANGUAGE_MODEL_COMPLETION (source_object),
                                                        &completed_string,
                                                        result,
                                                        &error))
    {
      g_task_return_error (task, g_steal_pointer (&error));
      return;
    }

  g_task_return_pointer (task, g_steal_pointer (&completed_string), g_free);
}

void
ggml_client_language_model_cursor_exec_async (GGMLClientLanguageModelCursor *cursor,
                                              size_t                         num_tokens,
                                              GCancellable                  *cancellable,
                                              GAsyncReadyCallback            callback,
                                              gpointer                       user_data)
{
  g_autoptr(CallCursorExecClosure) closure = call_cursor_exec_closure_new (cursor, callback, user_data);

  ggml_language_model_completion_call_exec (cursor->proxy,
                                            num_tokens,
                                            G_DBUS_CALL_FLAGS_NONE,
                                            -1,
                                            cancellable,
                                            on_done_complete_exec,
                                            g_steal_pointer (&closure));
}

char *
ggml_client_language_model_cursor_exec_finish (GGMLClientLanguageModelCursor  *cursor,
                                               GAsyncResult                   *result,
                                               GError                        **error)
{
  return g_task_propagate_pointer (G_TASK (result), error);
}

G_DEFINE_BOXED_TYPE (GGMLClientLanguageModelCursor,
                     ggml_client_language_model_cursor,
                     ggml_client_language_model_cursor_ref,
                     ggml_client_language_model_cursor_unref)

