/*
 * ggml-gobject/ggml-cached-model.c
 *
 * Library code for ggml-cached-model
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

#include <stdint.h>
#include <gio/gio.h>
#include <libsoup/soup.h>
#include <ggml-gobject/ggml-cached-model.h>
#include <ggml-gobject/internal/ggml-async-queue-source.h>
#include <ggml-gobject/internal/ggml-progress-istream.h>

struct _GGMLCachedModelIstream
{
  GFileInputStream parent_instance;
};

typedef struct
{
  GFileInputStream *current_stream;
  size_t remote_content_length;
  char *remote_url;
  char *local_path;
  uint32_t progress_indicator_source_id;
  GAsyncQueue *progress_indicator_queue;
  GFileProgressCallback progress_callback;
  gpointer progress_callback_data;
  GDestroyNotify progress_callback_data_destroy;
} GGMLCachedModelIstreamPrivate;

enum {
  PROP_0,
  PROP_REMOTE_URL,
  PROP_LOCAL_PATH,
  PROP_N
};

G_DEFINE_TYPE_WITH_CODE (GGMLCachedModelIstream,
                         ggml_cached_model_istream,
                         G_TYPE_FILE_INPUT_STREAM,
                         G_ADD_PRIVATE (GGMLCachedModelIstream))

static gboolean
ggml_download_progress_async_queue_monitor_callback (gpointer message,
                                                     gpointer user_data)
{
  GGMLCachedModelIstream *cached_model = user_data;
  GGMLCachedModelIstreamPrivate *priv = ggml_cached_model_istream_get_instance_private (cached_model);

  /* We progress the message by sending it to the progress callback */
  goffset progressed_bytes = GPOINTER_TO_INT (message);

  /* We unconditionally send to the progress callback, even if it is
   * the sentinel value - the progress callback has to be able to handle this */
  (*priv->progress_callback) (progressed_bytes,
                              (goffset) priv->remote_content_length,
                              priv->progress_callback_data);


  /* Sentinel message */
  if (progressed_bytes == -1)
    {
      return G_SOURCE_REMOVE;
    }

  return G_SOURCE_CONTINUE;
}

static void
ggml_cached_model_push_download_progress_to_queue (goffset progressed_bytes,
                                                   goffset total_bytes,
                                                   gpointer user_data)
{
  GGMLCachedModelIstream *cached_model = user_data;
  GGMLCachedModelIstreamPrivate *priv = ggml_cached_model_istream_get_instance_private (cached_model);

  /* Only push a message if one hasn't been consumed yet - otherwise
   * we run the risk of flooding the main progress */
  if (g_async_queue_length (priv->progress_indicator_queue) == 0)
    {
      g_async_queue_push (priv->progress_indicator_queue, GINT_TO_POINTER (progressed_bytes));
      g_main_context_wakeup (g_main_context_default ());
    }
}

/**
 * ggml_cached_model_istream_set_download_progress_callback:
 * @callback: A #GFileProgressCallback with progress about the download operation.
 * @user_data: (closure callback): A closure for @callback
 * @user_data_destroy: (destroy callback): A #GDestroyNotify for @callback
 *
 * Set a progress-monitor callback for @cached_model, which will be called with
 * download progress in case a model is being downloaded. The application can use
 * the callback to update state, for example a progress bar.
 *
 * This function should be called from the main thread. It will handle situations where
 * the download IO operation happens on a separate thread.
 */
void
ggml_cached_model_istream_set_download_progress_callback (GGMLCachedModelIstream *cached_model,
                                                          GFileProgressCallback   callback,
                                                          gpointer                user_data,
                                                          GDestroyNotify          user_data_destroy)
{
  GGMLCachedModelIstreamPrivate *priv = ggml_cached_model_istream_get_instance_private (cached_model);

  g_clear_pointer (&priv->progress_callback_data, priv->progress_callback_data_destroy);

  if (priv->progress_indicator_source_id != 0)
    {
      g_source_remove (priv->progress_indicator_source_id);
      priv->progress_indicator_source_id = 0;
    }

  g_clear_pointer (&priv->progress_indicator_queue, g_async_queue_unref);

  priv->progress_callback = callback;
  priv->progress_callback_data = user_data;
  priv->progress_callback_data_destroy = user_data_destroy;
}

static GFileInputStream *
ggml_cached_model_istream_ensure_stream (GGMLCachedModelIstream     *cached_model,
                                         GFileProgressCallback       progress_callback,
                                         gpointer                    progress_callback_data,
                                         GCancellable               *cancellable,
                                         GError                    **error)
{
  GGMLCachedModelIstreamPrivate *priv = ggml_cached_model_istream_get_instance_private (cached_model);

  /* We first try to read the local_path. But if that fails because
   * the file doesn't exist, then we have to download the model ourselves. */
  g_autoptr(GFile) local_file = g_file_new_for_path (priv->local_path);
  g_autoptr(GError) my_error = NULL;
  g_autoptr(GFileInputStream) local_file_stream = g_file_read (local_file, cancellable, &my_error);

  if (local_file_stream != NULL)
    {
      return g_steal_pointer (&local_file_stream);
    }

  g_assert (my_error != NULL);

  if (my_error->code != G_IO_ERROR_NOT_FOUND)
    {
      /* Some other error, eg, a directory is in the way, filesystem error,
       * etc. We can't do anything about it */
      g_propagate_error (error, g_steal_pointer (&my_error));
      return NULL;
    }

  g_clear_error (&my_error);

  /* In this case we have to download the file. Lets download first
   * into a temporary file, then move it into the correct place once
   * the download is done. */
  g_autoptr(GFileIOStream) io_stream = NULL;
  g_autoptr(GFile) output_file = g_file_new_tmp ("ggml-model-download-XXXXXX.bin", &io_stream, error);
  GOutputStream *output_stream = g_io_stream_get_output_stream (G_IO_STREAM (io_stream));

  if (output_stream == NULL)
    {
      return NULL;
    }

  g_autoptr(SoupSession) session = soup_session_new ();
  g_autoptr(SoupMessage) message = soup_message_new (SOUP_METHOD_GET, priv->remote_url);
  g_autoptr(GInputStream) in_stream = soup_session_send (session, message, cancellable, error);

  if (in_stream == NULL)
    {
      return NULL;
    }

  SoupMessageHeaders *response_headers = soup_message_get_response_headers (message);
  priv->remote_content_length = soup_message_headers_get_content_length (response_headers);

  g_autoptr(GGMLProgressIstream) progress_istream = ggml_progress_istream_new (in_stream,
                                                                               priv->remote_content_length);

  if (priv->progress_callback != NULL)
    {
      priv->progress_indicator_queue = g_async_queue_new ();

      GSource *monitor_source = ggml_async_queue_source_new (priv->progress_indicator_queue,
                                                             ggml_download_progress_async_queue_monitor_callback,
                                                             g_object_ref (cached_model),
                                                             (GDestroyNotify) g_object_unref,
                                                             cancellable);
      g_source_attach (g_steal_pointer (&monitor_source), NULL);

      ggml_progress_istream_set_callback (progress_istream,
                                          ggml_cached_model_push_download_progress_to_queue,
                                          g_object_ref (cached_model),
                                          g_object_unref);
    }

  if (g_output_stream_splice (G_OUTPUT_STREAM (output_stream),
                              G_INPUT_STREAM (progress_istream),
                              G_OUTPUT_STREAM_SPLICE_CLOSE_SOURCE |
                              G_OUTPUT_STREAM_SPLICE_CLOSE_TARGET,
                              cancellable,
                              error) == -1)
    {
      /* We send the sentinel value to the progress callback on the error
       * case too, so that it can clean up */
      if (priv->progress_callback != NULL)
        {
          ggml_cached_model_push_download_progress_to_queue (-1,
                                                             priv->remote_content_length,
                                                             cached_model);
        }

      return NULL;
    }

  /* Once we're done, send the sentinel message to the queue */
  if (priv->progress_callback != NULL)
    {
      ggml_cached_model_push_download_progress_to_queue (-1,
                                                         priv->remote_content_length,
                                                         cached_model);
    }

  /* After that, we have to move the temporary file into the right place. */
  g_autoptr(GFile) output_directory = g_file_get_parent (local_file);

  if (!g_file_make_directory_with_parents (output_directory, cancellable, &my_error))
    {
      if (my_error->code != G_IO_ERROR_EXISTS)
        {
          g_propagate_error (error, g_steal_pointer (&my_error));
          return NULL;
        }
      g_clear_error (&my_error);
    }

  if (!g_file_move (output_file,
                    local_file,
                    G_FILE_COPY_OVERWRITE,
                    cancellable,
                    NULL,
                    NULL,
                    error))
    {
      return NULL;
    }

  /* We call the same function again, now that the cached file is in place. */
  return ggml_cached_model_istream_ensure_stream (cached_model,
                                                  progress_callback,
                                                  progress_callback_data,
                                                  cancellable,
                                                  error);
}

static void
ggml_cached_model_istream_dispose (GObject *object)
{
  GGMLCachedModelIstream *cached_model = GGML_CACHED_MODEL_ISTREAM (object);
  GGMLCachedModelIstreamPrivate *priv = ggml_cached_model_istream_get_instance_private (cached_model);


  g_clear_pointer (&priv->progress_indicator_queue, g_async_queue_unref);
  g_clear_pointer (&priv->progress_callback_data, priv->progress_callback_data_destroy);

  /* If for some reason the source is still there, drop it */
  if (priv->progress_indicator_source_id != 0)
    {
      g_source_remove (priv->progress_indicator_source_id);
      priv->progress_indicator_source_id = 0;
    }

  G_OBJECT_CLASS (ggml_cached_model_istream_parent_class)->finalize (object);
}

static void
ggml_cached_model_istream_finalize (GObject *object)
{
  GGMLCachedModelIstream *cached_model = GGML_CACHED_MODEL_ISTREAM (object);
  GGMLCachedModelIstreamPrivate *priv = ggml_cached_model_istream_get_instance_private (cached_model);

  g_clear_pointer (&priv->local_path, g_free);
  g_clear_pointer (&priv->remote_url, g_free);

  G_OBJECT_CLASS (ggml_cached_model_istream_parent_class)->finalize (object);
}

static void
ggml_cached_model_istream_get_property (GObject    *object,
                                        uint32_t    property_id,
                                        GValue     *value,
                                        GParamSpec *pspec)
{
  GGMLCachedModelIstream *cached_model = GGML_CACHED_MODEL_ISTREAM (object);
  GGMLCachedModelIstreamPrivate *priv = ggml_cached_model_istream_get_instance_private (cached_model);

  switch (property_id)
    {
      case PROP_REMOTE_URL:
        g_value_set_string (value, priv->remote_url);
        break;
      case PROP_LOCAL_PATH:
        g_value_set_string (value, priv->local_path);
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
        break;
    }
}

static void
ggml_cached_model_istream_set_property (GObject      *object,
                                        uint32_t      property_id,
                                        const GValue *value,
                                        GParamSpec   *pspec)
{
  GGMLCachedModelIstream *cached_model = GGML_CACHED_MODEL_ISTREAM (object);
  GGMLCachedModelIstreamPrivate *priv = ggml_cached_model_istream_get_instance_private (cached_model);

  switch (property_id)
    {
      case PROP_REMOTE_URL:
        g_clear_pointer (&priv->remote_url, g_free);
        priv->remote_url = g_value_dup_string (value);
        break;
      case PROP_LOCAL_PATH:
        g_clear_pointer (&priv->local_path, g_free);
        priv->local_path = g_value_dup_string (value);
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
        break;
    }
}

static ssize_t
ggml_cached_model_istream_read_fn (GInputStream  *stream,
                                   void          *buffer,
                                   gsize          count,
                                   GCancellable  *cancellable,
                                   GError       **error)
{
  GGMLCachedModelIstream *cached_model = GGML_CACHED_MODEL_ISTREAM (stream);
  GGMLCachedModelIstreamPrivate *priv = ggml_cached_model_istream_get_instance_private (cached_model);

  if (priv->current_stream == NULL)
    {
      priv->current_stream = ggml_cached_model_istream_ensure_stream (cached_model,
                                                                      priv->progress_callback,
                                                                      priv->progress_callback_data,
                                                                      cancellable,
                                                                      error);

      if (priv->current_stream == NULL)
        {
          return -1;
        }
    }

  return g_input_stream_read (G_INPUT_STREAM (priv->current_stream), buffer, count, cancellable, error);
}

static ssize_t
ggml_cached_model_istream_skip (GInputStream  *stream,
                                size_t         count,
                                GCancellable  *cancellable,
                                GError       **error)
{
  GGMLCachedModelIstream *cached_model = GGML_CACHED_MODEL_ISTREAM (stream);
  GGMLCachedModelIstreamPrivate *priv = ggml_cached_model_istream_get_instance_private (cached_model);

  if (priv->current_stream == NULL)
    {
      priv->current_stream = ggml_cached_model_istream_ensure_stream (cached_model,
                                                                      priv->progress_callback,
                                                                      priv->progress_callback_data,
                                                                      cancellable,
                                                                      error);

      if (priv->current_stream == NULL)
        {
          return -1;
        }
    }

  return g_input_stream_skip (G_INPUT_STREAM (priv->current_stream), count, cancellable, error);
}

static gboolean
ggml_cached_model_istream_close (GInputStream  *stream,
                                 GCancellable  *cancellable,
                                 GError       **error)
{
  GGMLCachedModelIstream *cached_model = GGML_CACHED_MODEL_ISTREAM (stream);
  GGMLCachedModelIstreamPrivate *priv = ggml_cached_model_istream_get_instance_private (cached_model);

  if (priv->current_stream == NULL)
    {
      /* Do nothing, we don't have the stream open anyway */
      return TRUE;
    }

  return g_input_stream_close (G_INPUT_STREAM (priv->current_stream), cancellable, error);
}

static goffset
ggml_cached_model_istream_tell (GFileInputStream *file_istream)
{
  GGMLCachedModelIstream *cached_model = GGML_CACHED_MODEL_ISTREAM (file_istream);
  GGMLCachedModelIstreamPrivate *priv = ggml_cached_model_istream_get_instance_private (cached_model);
  g_autoptr(GError) error = NULL;

  if (priv->current_stream == NULL)
    {
      return 0;
    }

  return g_seekable_tell (G_SEEKABLE (priv->current_stream));
}

static gboolean
ggml_cached_model_istream_can_seek (GFileInputStream *file_istream)
{
  return FALSE;
}

static GFileInfo *
ggml_cached_model_istream_query_info (GFileInputStream  *stream,
                                      const char        *attributes,
                                      GCancellable      *cancellable,
                                      GError           **error)
{
  GGMLCachedModelIstream *cached_model = GGML_CACHED_MODEL_ISTREAM (stream);
  GGMLCachedModelIstreamPrivate *priv = ggml_cached_model_istream_get_instance_private (cached_model);

  /* Downloading the whole file just to get the attributes is daft,
   * but that's the only way to get what the user is asking for. */
  if (priv->current_stream == NULL)
    {
      priv->current_stream = ggml_cached_model_istream_ensure_stream (cached_model,
                                                                      priv->progress_callback,
                                                                      priv->progress_callback_data,
                                                                      cancellable,
                                                                      error);

      if (priv->current_stream == NULL)
        {
          return NULL;
        }
    }

  return g_file_input_stream_query_info (priv->current_stream,
                                         attributes,
                                         cancellable,
                                         error);
}

static void
ggml_cached_model_istream_init (GGMLCachedModelIstream *cached_model)
{
  GGMLCachedModelIstreamPrivate *priv = ggml_cached_model_istream_get_instance_private (cached_model);

  priv->current_stream = NULL;
}

static void
ggml_cached_model_istream_class_init (GGMLCachedModelIstreamClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GInputStreamClass *stream_class = G_INPUT_STREAM_CLASS (klass);
  GFileInputStreamClass *file_stream_class = G_FILE_INPUT_STREAM_CLASS (klass);

  object_class->finalize = ggml_cached_model_istream_finalize;
  object_class->dispose = ggml_cached_model_istream_dispose;
  object_class->set_property = ggml_cached_model_istream_set_property;
  object_class->get_property = ggml_cached_model_istream_get_property;

  stream_class->read_fn = ggml_cached_model_istream_read_fn;
  stream_class->skip = ggml_cached_model_istream_skip;
  stream_class->close_fn = ggml_cached_model_istream_close;

  file_stream_class->tell = ggml_cached_model_istream_tell;
  file_stream_class->can_seek = ggml_cached_model_istream_can_seek;
  file_stream_class->query_info = ggml_cached_model_istream_query_info;

  g_object_class_install_property (object_class,
                                   PROP_REMOTE_URL,
                                   g_param_spec_string ("remote-url",
                                                        "Remote URL",
                                                        "Remote URL",
                                                        NULL,
                                                        G_PARAM_READWRITE |
                                                        G_PARAM_CONSTRUCT));
  g_object_class_install_property (object_class,
                                   PROP_LOCAL_PATH,
                                   g_param_spec_string ("local-path",
                                                        "Local Path",
                                                        "Local Path",
                                                        NULL,
                                                        G_PARAM_READWRITE |
                                                        G_PARAM_CONSTRUCT));
}

GGMLCachedModelIstream *
ggml_cached_model_istream_new (const char *remote_url,
                               const char *local_path)
{
  return GGML_CACHED_MODEL_ISTREAM (g_object_new (GGML_TYPE_CACHED_MODEL_ISTREAM,
                                                  "remote-url", remote_url,
                                                  "local-path", local_path,
                                                  NULL));
}
