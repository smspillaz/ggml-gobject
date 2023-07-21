/*
 * ggml-gobject/ggml-progress-istream.c
 *
 * Library code for ggml-progress-istream
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
#include <ggml-gobject/internal/ggml-progress-istream.h>

struct _GGMLProgressIstream
{
  GFilterInputStream parent_instance;
};

typedef struct
{
  size_t bytes_consumed;
  size_t expected_size;
  GFileProgressCallback progress_callback;
  gpointer progress_callback_data;
  GDestroyNotify progress_callback_data_destroy;
} GGMLProgressIstreamPrivate;

enum {
  PROP_0,
  PROP_EXPECTED_SIZE,
  PROP_N
};

G_DEFINE_TYPE_WITH_CODE (GGMLProgressIstream,
                         ggml_progress_istream,
                         G_TYPE_FILTER_INPUT_STREAM,
                         G_ADD_PRIVATE (GGMLProgressIstream))

void
ggml_progress_istream_set_callback (GGMLProgressIstream   *istream,
                                    GFileProgressCallback  callback,
                                    gpointer               user_data,
                                    GDestroyNotify         user_data_destroy)
{
  GGMLProgressIstream *progress_istream = istream;
  GGMLProgressIstreamPrivate *priv = ggml_progress_istream_get_instance_private (progress_istream);

  g_clear_pointer (&priv->progress_callback_data, priv->progress_callback_data_destroy);

  priv->progress_callback = callback;
  priv->progress_callback_data = user_data;
  priv->progress_callback_data_destroy = user_data_destroy;
}

static void
ggml_progress_istream_dispose (GObject *object)
{
  GGMLProgressIstream *progress_istream = GGML_PROGRESS_ISTREAM (object);
  GGMLProgressIstreamPrivate *priv = ggml_progress_istream_get_instance_private (progress_istream);

  g_clear_pointer (&priv->progress_callback_data, priv->progress_callback_data_destroy);

  G_OBJECT_CLASS (ggml_progress_istream_parent_class)->dispose (object);
}

static void
ggml_progress_istream_get_property (GObject    *object,
                                    uint32_t    property_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  GGMLProgressIstream *progress_istream = GGML_PROGRESS_ISTREAM (object);
  GGMLProgressIstreamPrivate *priv = ggml_progress_istream_get_instance_private (progress_istream);

  switch (property_id)
    {
      case PROP_EXPECTED_SIZE:
        g_value_set_uint (value, priv->expected_size);
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
        break;
    }
}

static void
ggml_progress_istream_set_property (GObject      *object,
                                    uint32_t      property_id,
                                    const GValue *value,
                                    GParamSpec   *pspec)
{
  GGMLProgressIstream *progress_istream = GGML_PROGRESS_ISTREAM (object);
  GGMLProgressIstreamPrivate *priv = ggml_progress_istream_get_instance_private (progress_istream);

  switch (property_id)
    {
      case PROP_EXPECTED_SIZE:
        priv->expected_size = g_value_get_uint (value);
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
        break;
    }
}

static ssize_t
ggml_progress_istream_read_fn (GInputStream  *stream,
                               void          *buffer,
                               gsize          count,
                               GCancellable  *cancellable,
                               GError       **error)
{
  GGMLProgressIstream *progress_istream = GGML_PROGRESS_ISTREAM (stream);
  GGMLProgressIstreamPrivate *priv = ggml_progress_istream_get_instance_private (progress_istream);

  ssize_t result = G_INPUT_STREAM_CLASS (ggml_progress_istream_parent_class)->read_fn (stream, buffer, count, cancellable, error);

  /* Report back to the progress callback that we read result bytes */
  if (result != -1)
    {
      priv->bytes_consumed += result;

      if (priv->progress_callback != NULL)
        {
          (*priv->progress_callback) (priv->bytes_consumed,
                                      priv->expected_size,
                                      priv->progress_callback_data);
        }
    }

  return result;
}

static ssize_t
ggml_progress_istream_skip (GInputStream  *stream,
                            size_t         count,
                            GCancellable  *cancellable,
                            GError       **error)
{
  GGMLProgressIstream *progress_istream = GGML_PROGRESS_ISTREAM (stream);
  GGMLProgressIstreamPrivate *priv = ggml_progress_istream_get_instance_private (progress_istream);

  ssize_t result = G_INPUT_STREAM_CLASS (ggml_progress_istream_parent_class)->skip (stream, count, cancellable, error);

  /* Report back to the progress callback that we read result bytes */
  if (result != -1)
    {
      priv->bytes_consumed += result;

      if (priv->progress_callback != NULL)
        {
          (*priv->progress_callback) (priv->bytes_consumed,
                                      priv->expected_size,
                                      priv->progress_callback_data);
        }
    }

  return result;
}

static void
ggml_progress_istream_init (GGMLProgressIstream *progress_istream)
{
}

static void
ggml_progress_istream_class_init (GGMLProgressIstreamClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GInputStreamClass *stream_class = G_INPUT_STREAM_CLASS (klass);

  object_class->dispose = ggml_progress_istream_dispose;
  object_class->set_property = ggml_progress_istream_set_property;
  object_class->get_property = ggml_progress_istream_get_property;

  stream_class->read_fn = ggml_progress_istream_read_fn;
  stream_class->skip = ggml_progress_istream_skip;

  g_object_class_install_property (object_class,
                                   PROP_EXPECTED_SIZE,
                                   g_param_spec_uint ("expected-size",
                                                      "Expected Size",
                                                      "Expected Size",
                                                      1,
                                                      G_MAXUINT,
                                                      1,
                                                      G_PARAM_READWRITE |
                                                      G_PARAM_CONSTRUCT));
}

GGMLProgressIstream *
ggml_progress_istream_new (GInputStream *base_stream,
                           size_t        expected_size)
{
  return GGML_PROGRESS_ISTREAM (g_object_new (GGML_TYPE_PROGRESS_ISTREAM,
                                              "base-stream", base_stream,
                                              "expected-size", expected_size,
                                              NULL));
}
