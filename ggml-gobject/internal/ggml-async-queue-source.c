/*
 * ggml-gobject/ggml-async-queue-source.c
 *
 * Library code for ggml-async-queue-source
 *
 * Copyright (C) 2023 Sam Spilsbury.
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
#include <ggml-gobject/internal/ggml-async-queue-source.h>

typedef struct {
  GSource source;
  GAsyncQueue *queue;
} GGMLAsyncQueueSource;

static gboolean
ggml_async_queue_source_prepare (GSource *source, int32_t *timeout)
{
  GGMLAsyncQueueSource *async_queue_source = (GGMLAsyncQueueSource *) source;
  return g_async_queue_length (async_queue_source->queue) > 0;
}

static gboolean
ggml_async_queue_source_dispatch (GSource     *source,
                                  GSourceFunc  func,
                                  gpointer     user_data)
{
  GGMLAsyncQueueSource *async_queue_source = (GGMLAsyncQueueSource *) source;
  gpointer message = g_async_queue_try_pop (async_queue_source->queue);
  GGMLAsyncQueueSourceDispatchFunc real_func = (GGMLAsyncQueueSourceDispatchFunc) func;

  if (message != NULL)
    {
      g_assert (real_func != NULL);
      return real_func (message, user_data);
    }

  return G_SOURCE_CONTINUE;
}

static void
ggml_async_queue_source_finalize (GSource *source)
{
  GGMLAsyncQueueSource *async_queue_source = (GGMLAsyncQueueSource *) source;

  g_clear_pointer (&async_queue_source->queue, g_async_queue_unref);

  /* The source will be freed later */
}

static GSourceFuncs async_queue_source_funcs = {
  .prepare = ggml_async_queue_source_prepare,
  .check = NULL,
  .dispatch = ggml_async_queue_source_dispatch,
  .finalize = ggml_async_queue_source_finalize
};

/**
 * ggml_async_queue_source_new:
 * @queue: A #GAsyncQueue
 * @func: A #GGMLAsyncQueueSourceDispatchFunc
 * @user_data: (closure @func): A closure for @func
 * @user_data_destroy: (destroy @func): A destructor for @user_data
 * @cancellable: A #GCancellable
 *
 * Create a new #GGMLAsyncQueueSource from #GAsyncQueue . The source will
 * be dispatched when there is work to be done from the @queue. The queue
 * needs to wakeup the main context manually by calling g_main_context_wakeup
 * once the source is ready to be dispatched, as there is no poll-filedescriptor
 */
GSource *
ggml_async_queue_source_new (GAsyncQueue                      *queue,
                             GGMLAsyncQueueSourceDispatchFunc  func,
                             gpointer                          user_data,
                             GDestroyNotify                    user_data_destroy,
                             GCancellable                     *cancellable)
{
  
  GGMLAsyncQueueSource *source = (GGMLAsyncQueueSource *) g_source_new (&async_queue_source_funcs, sizeof (GGMLAsyncQueueSource));

  g_source_set_callback ((GSource *) source, (GSourceFunc) func, user_data, user_data_destroy);
  g_source_set_name ((GSource *) source, "AsyncQueueSource");

  source->queue = g_async_queue_ref (queue);

  if (cancellable != NULL)
    {
      g_autoptr(GSource) cancellable_source = g_cancellable_source_new (cancellable);
      g_source_set_dummy_callback (cancellable_source);
      g_source_add_child_source ((GSource *) source, cancellable_source);
    }

  return (GSource *) source;
}
