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

static GFileInputStream *
ggml_cached_model_istream_ensure_stream (GGMLCachedModelIstream  *cached_model,
                                         GCancellable            *cancellable,
                                         GError                 **error)
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

  if (!g_output_stream_splice (G_OUTPUT_STREAM (output_stream),
                               in_stream,
                               G_OUTPUT_STREAM_SPLICE_CLOSE_SOURCE |
                               G_OUTPUT_STREAM_SPLICE_CLOSE_TARGET,
                               cancellable,
                               error))
    {
      return NULL;
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
  return ggml_cached_model_istream_ensure_stream (cached_model, cancellable, error);
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
ggml_cached_model_istream_dispose (GObject *object)
{
  G_OBJECT_CLASS (ggml_cached_model_istream_parent_class)->dispose (object);
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
      priv->current_stream = ggml_cached_model_istream_ensure_stream (cached_model, cancellable, error);

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
      priv->current_stream = ggml_cached_model_istream_ensure_stream (cached_model, cancellable, error);

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
      priv->current_stream = ggml_cached_model_istream_ensure_stream (cached_model, cancellable, error);

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

GFileInputStream *
ggml_cached_model_istream_new (const char *remote_url,
                               const char *local_path)
{
  return G_FILE_INPUT_STREAM (g_object_new (GGML_TYPE_CACHED_MODEL_ISTREAM,
                                            "remote-url", remote_url,
                                            "local-path", local_path,
                                            NULL));
}