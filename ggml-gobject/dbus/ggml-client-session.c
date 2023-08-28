/*
 * ggml-gobject/dbus/ggml-client-session.c
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

#include <ggml-gobject/dbus/ggml-service-dbus.h>
#include <ggml-gobject/dbus/ggml-client-service.h>
#include <ggml-gobject/dbus/ggml-client-session.h>
#include <ggml-gobject/dbus/internal/ggml-client-session-internal.h>
#include <ggml-gobject/dbus/internal/ggml-client-language-model-cursor-internal.h>

struct _GGMLClientSession {
  GGMLSession *proxy;
  size_t ref_count;
};

GGMLClientSession *
ggml_client_session_new (GGMLSession *proxy)
{
  GGMLClientSession *session_client = g_new0 (GGMLClientSession, 1);
  session_client->proxy = g_object_ref (proxy);

  session_client->ref_count = 1;
  return session_client;
}

/**
 * ggml_client_session_ref: (skip):
 * @session_client: A #GGMLClientSession
 *
 * Increase ref count on this #GGMLClientSession
 *
 * Returns: (transfer full): A #GGMLClientSession
 */
GGMLClientSession *
ggml_client_session_ref (GGMLClientSession *session_client)
{
  ++session_client->ref_count;

  return session_client;
}

/**
 * ggml_client_session_unref: (skip):
 * @session_client: A #GGMLClientSession
 *
 * Decreases the ref count on @session_client. If the ref count drops
 * to zero, then @session_client will be freed
 */
void
ggml_client_session_unref (GGMLClientSession *session_client)
{
  if (--session_client->ref_count == 0)
    {
      g_clear_object (&session_client->proxy);
      g_clear_pointer (&session_client, g_free);
    }
}

typedef struct {
  GCancellable *cancellable;
  GGMLClientSession *session_client;
  GAsyncReadyCallback callback;
  gpointer user_data;
} CallCreateCompletionClosure;

static CallCreateCompletionClosure *
call_create_completion_closure_new (GGMLClientSession                          *session_client,
                                    GCancellable                               *cancellable,
                                    GAsyncReadyCallback                         callback,
                                    gpointer                                    user_data)
{
  CallCreateCompletionClosure *closure = g_new0 (CallCreateCompletionClosure, 1);
  closure->session_client = ggml_client_session_ref (session_client);
  closure->cancellable = cancellable != NULL ? g_object_ref (cancellable) : NULL;
  closure->callback = callback;
  closure->user_data = user_data;

  return closure;
}

static void
call_create_completion_closure_free (CallCreateCompletionClosure *closure)
{
  g_clear_object (&closure->cancellable);
  g_clear_pointer (&closure->session_client, ggml_client_session_unref);
  g_clear_pointer (&closure, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (CallCreateCompletionClosure, call_create_completion_closure_free)

static void
on_completion_object_ready (GObject       *source_object,
                            GAsyncResult  *result,
                            gpointer       user_data)
{
  g_autoptr(CallCreateCompletionClosure) closure = user_data;
  g_autoptr(GError) error = NULL;
  g_autoptr(GTask) task = g_task_new (NULL, NULL, closure->callback, closure->user_data);
  g_autoptr(GGMLLanguageModelCompletion) completion_proxy = ggml_language_model_completion_proxy_new_finish (result, &error);

  if (completion_proxy == NULL)
    {
      g_error ("Failed to create LanguageModelCompletion proxy: %s", error->message);
      g_task_return_error (task, g_steal_pointer (&error));
      return;
    }

  g_autoptr(GGMLClientLanguageModelCursor) cursor_client = ggml_client_language_model_cursor_new (completion_proxy);
  g_task_return_pointer (task, g_steal_pointer (&cursor_client), (GDestroyNotify) ggml_client_language_model_cursor_unref);
}

static void
on_call_create_completion_reply (GObject      *source_object,
                                 GAsyncResult *result,
                                 gpointer      user_data)
{
  g_autoptr(CallCreateCompletionClosure) closure = user_data;
  g_autoptr(GError) error = NULL;
  g_autofree char *completion_object_path = NULL;

  if (!ggml_session_call_create_completion_finish (closure->session_client->proxy,
                                                   &completion_object_path,
                                                   result,
                                                   &error))
    {
      g_autoptr(GTask) task = g_task_new (NULL, NULL, closure->callback, closure->user_data);
      g_error ("Failed to create completion: %s\n", error->message);
      g_task_return_error (task, g_steal_pointer (&error));
      return;
    }

  /* Now we have to create a proxy for the object path on the bus */
  ggml_language_model_completion_proxy_new (
    g_dbus_proxy_get_connection (G_DBUS_PROXY (closure->session_client->proxy)),
    G_DBUS_PROXY_FLAGS_NONE,
    NULL,
    completion_object_path,
    closure->cancellable,
    on_completion_object_ready,
    g_steal_pointer (&closure)
  );
}

/**
 * ggml_client_session_start_completion_async:
 * @session_client: A #GGMLSessionClient
 * @model_name: The model name to use
 * @model_variant: (nullable): The model variant to use
 * @quantization: (nullable): The quantization level to use
 * @prompt: The initial prompt to start the model with
 * @max_tokens: Maximum number of tokens to be generated
 * @properties: A #GVariant of additional properties to pass
 * @cancellable: (nullable): A #GCancellable
 * @callback: A #GAsyncReadyCallback to be called when the completion object is ready
 * @user_data: (closure callback): User data for @callback.
 *
 * Asynchronously created a #GGMLClientLanguageModelCursor object, calling @callback
 * when the object is ready. Complete the call with %ggml_client_session_start_completion_finish .
 */
void
ggml_client_session_start_completion_async (GGMLClientSession                          *session_client,
                                            const char                                 *model_name,
                                            const char                                 *model_variant,
                                            const char                                 *quantization,
                                            const char                                 *prompt,
                                            size_t                                      max_tokens,
                                            GVariant                                   *properties,
                                            GCancellable                               *cancellable,
                                            GAsyncReadyCallback                         callback,
                                            gpointer                                    user_data)
{
  g_autoptr(CallCreateCompletionClosure) closure = call_create_completion_closure_new (session_client,
                                                                                       cancellable,
                                                                                       callback,
                                                                                       user_data);
  GVariantBuilder builder;
  g_variant_builder_init (&builder, G_VARIANT_TYPE_ARRAY);
  g_variant_builder_add (&builder, "{sv}", "n_params", g_variant_new_string (model_variant));
  g_variant_builder_add (&builder, "{sv}", "quantization", g_variant_new_string (quantization));

  if (properties != NULL)
    {
      GVariantIter iter;
      g_variant_iter_init (&iter, properties);

      char *key;
      GVariant *value;

      while (g_variant_iter_loop (&iter, "{sv}", &key, &value))
        {
          g_variant_builder_add (&builder, "{sv}", key, value);
        }
    }

  g_autoptr(GVariant) completion_properties = g_variant_ref_sink (g_variant_builder_end (&builder));

  /* Now lets create a cursor and start doing some inference */
  ggml_session_call_create_completion (session_client->proxy,
                                       model_name,
                                       completion_properties,
                                       prompt,
                                       max_tokens,
                                       G_DBUS_CALL_FLAGS_NONE,
                                       -1,
                                       cancellable,
                                       on_call_create_completion_reply,
                                       g_steal_pointer (&closure));
}

/**
 * ggml_client_session_start_completion_finish:
 * @result: A #GAsyncResult
 * @error: A #GError
 *
 * Completes the call to %ggml_client_session_start_completion_async .
 *
 * Returns: (transfer full): A #GGMLClientLanguageModelCursor on success or
 *          %NULL with @error set on failure.
 */
GGMLClientLanguageModelCursor *
ggml_client_session_start_completion_finish (GAsyncResult  *result,
                                             GError       **error)
{
  return g_task_propagate_pointer (G_TASK (result), error);
}

typedef struct {
  GCancellable        *cancellable;
  GAsyncReadyCallback  callback;
  gpointer             user_data;
} CallNewServiceClosure;

static CallNewServiceClosure *
call_new_service_closure_new (GCancellable       *cancellable,
                              GAsyncReadyCallback callback,
                              gpointer            user_data)
{
  CallNewServiceClosure *closure = g_new0 (CallNewServiceClosure, 1);
  closure->cancellable = cancellable != NULL ? g_object_ref (cancellable) : NULL;
  closure->callback = callback;
  closure->user_data = user_data;

  return closure;
}

static void
call_new_service_closure_free (CallNewServiceClosure *closure)
{
  g_clear_object (&closure->cancellable);
  g_clear_pointer (&closure, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (CallNewServiceClosure, call_new_service_closure_free);

static void
on_client_open_session (GObject      *source_object,
                        GAsyncResult *result,
                        gpointer      user_data)
{
  g_autoptr(CallNewServiceClosure) closure = user_data;
  g_autoptr(GTask) task = g_task_new (NULL, NULL, closure->callback, closure->user_data);
  g_autoptr(GError) error = NULL;
  g_autoptr(GGMLClientSession) session = ggml_client_service_open_session_finish (result, &error);

  if (session == NULL)
    {
      g_task_return_error (task, g_steal_pointer (&error));
      return;
    }

  g_task_return_pointer (task, g_steal_pointer (&session), (GDestroyNotify) ggml_client_session_unref);
}

static void
on_created_service (GObject      *source_object,
                    GAsyncResult *result,
                    gpointer      user_data)
{
  g_autoptr(CallNewServiceClosure) closure = user_data;
  g_autoptr(GError) error = NULL;
  g_autoptr(GGMLClientService) client = ggml_client_service_new_finish (result, &error);

  if (client == NULL)
    {
      g_autoptr(GTask) task = g_task_new (NULL, NULL, closure->callback, closure->user_data);
      g_task_return_error (task, g_steal_pointer (&error));
      return;
    }

  GCancellable *cancellable = closure->cancellable;

  /* Now we can open the session */
  ggml_client_service_open_session_async (client,
                                          cancellable,
                                          on_client_open_session,
                                          g_steal_pointer (&closure));
}

/** 
 * ggml_client_session_new_async:
 * @cancellable: A #GCancellable
 * @callback: A #GAsyncReadyCallback
 * @user_data: (closure callback): User data for @callback
 *
 * Asynchronously open a new #GGMLClientSession . Once the session is ready
 * @callback will be called. Complete the call with %ggml_client_new_session_async
 */
void
ggml_client_session_new_async (GCancellable        *cancellable,
                               GAsyncReadyCallback  callback,
                               gpointer             user_data)
{
  g_autoptr(CallNewServiceClosure) closure = call_new_service_closure_new (cancellable,
                                                                           callback,
                                                                           user_data);

  ggml_client_service_new_async (cancellable,
                                 on_created_service,
                                 g_steal_pointer (&closure));
}

/** 
 * ggml_client_new_session_finish:
 * @result: A #GAsyncResult
 * @error: A #GError out-parameter
 *
 * Completes the call to %ggml_client_session_new_async
 *
 * Returns: (transfer full): A new #GGMLClientSession or %NULL with
 *          @error set on failure.
 */
GGMLClientSession *
ggml_client_session_new_finish (GAsyncResult  *result,
                                GError       **error)
{
  return g_task_propagate_pointer (G_TASK (result), error);
}

G_DEFINE_BOXED_TYPE (GGMLClientSession,
                     ggml_client_session,
                     ggml_client_session_ref,
                     ggml_client_session_unref);
