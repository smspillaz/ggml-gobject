/*
 * ggml-gobject/ggml-async-queue-source.h
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

#include <glib-object.h>

G_BEGIN_DECLS

/**
 * GGMLAsyncQueueSourceDispatchFunc:
 * @item: Some item to be dispatched in the queue
 * @user_data: User data for the dispatch func
 *
 * Returns: A boolean for whether the source should be removed,
 *          usually this is either %G_SOURCE_REMOVE or %G_SOURCE_CONTINUE.
 */
typedef gboolean (*GGMLAsyncQueueSourceDispatchFunc) (gpointer item, gpointer user_data);

GSource *
ggml_async_queue_source_new (GAsyncQueue                      *queue,
                             GGMLAsyncQueueSourceDispatchFunc  func,
                             gpointer                          user_data,
                             GDestroyNotify                    user_data_destroy,
                             GCancellable                     *cancellable);

G_END_DECLS
