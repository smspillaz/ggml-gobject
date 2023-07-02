/*
 * ggml-gobject/ggml-hyperparameters.c
 *
 * Library code for ggml-hyperparameters
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

#include <ggml-gobject/ggml-hyperparameters.h>
#include <ggml-gobject/internal/ggml-stream-internal.h>

struct _GGMLHyperparameters {
  gchar **ordered_keys;
  GHashTable *parameters;
  size_t ref_count;
};

/**
 * ggml_hyperparameters_new:
 * @ordered_keys: (array zero-terminated=1): The hyperparameter names
 * @ordered_values: (array length=n_ordered_values): The hyperparameter values.
 * @n_ordered_values: Number of hyperparameter_values
 */
GGMLHyperparameters *
ggml_hyperparameters_new (const char **ordered_keys, int *ordered_values, size_t n_ordered_values)
{
  GGMLHyperparameters *parameters = g_new0 (GGMLHyperparameters, 1);

  g_assert (g_strv_length ((char **) ordered_keys) == n_ordered_values);

  parameters->ordered_keys = g_strdupv ((char **) ordered_keys);
  parameters->parameters = g_hash_table_new_full (g_str_hash, g_str_equal, NULL, NULL);
  parameters->ref_count = 1;

  const char **ordered_keys_iterator = ordered_keys;
  int *ordered_values_iterator = ordered_values;

  while (*ordered_keys_iterator != NULL)
    {
      g_hash_table_insert (parameters->parameters, (gpointer) *(ordered_keys_iterator++), GINT_TO_POINTER (*(ordered_values_iterator++)));
    }

  return parameters;
}

/**
 * ggml_hyperparameters_from_load_istream:
 * @istream: (transfer none): A #GInputStream
 * @cancellable: (transfer none): A #GCancellable
 * @error: A #GError out variable
 *
 * Returns: (transfer full): A #GGMLHyperparameters loaded from @istream or %NULL
 *          with @error set on failure.
 */
GGMLHyperparameters *
ggml_hyperparameters_load_from_istream (GInputStream *istream,
                                        GCancellable *cancellable,
                                        GError **error)
{
  int32_t parameter_values[6];

  if (!ggml_input_stream_read_exactly (istream, (char *) parameter_values, sizeof (int32_t) * 6, cancellable, error))
    {
      return NULL;
    }

  const char *parameter_keys[] = {
    "n_vocab",
    "n_ctx",
    "n_embd",
    "n_head",
    "n_layer",
    "ftype",
    NULL
  };

  return ggml_hyperparameters_new (parameter_keys, parameter_values, 6);
}

typedef struct _GGMLHyperparametersLoadFromIstreamData
{
  GInputStream *istream;
} GGMLHyperparametersLoadFromIstreamData;

static GGMLHyperparametersLoadFromIstreamData *
ggml_hyperparameters_load_from_istream_data_new (GInputStream *istream)
{
  GGMLHyperparametersLoadFromIstreamData *data = g_new0 (GGMLHyperparametersLoadFromIstreamData, 1);
  data->istream = g_object_ref (istream);

  return data;
}

static void
ggml_hyperparameters_load_from_istream_data_free (GGMLHyperparametersLoadFromIstreamData *data)
{
  g_clear_pointer (&data->istream, g_object_unref);
  g_clear_pointer (&data, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLHyperparametersLoadFromIstreamData, ggml_hyperparameters_load_from_istream_data_free);

static void
ggml_hyperparameters_load_from_istream_async_thread (GTask         *task,
                                                     gpointer       source_object,
                                                     gpointer       task_data,
                                                     GCancellable  *cancellable)
{
  GGMLHyperparametersLoadFromIstreamData *data = task_data;
  GError *error = NULL;

  g_autoptr(GGMLHyperparameters) hyperparameters = ggml_hyperparameters_load_from_istream (data->istream,
                                                                                           cancellable,
                                                                                           &error);

  if (hyperparameters == NULL)
    {
      g_task_return_error (task, error);
    }

  g_task_return_pointer (task, g_steal_pointer (&hyperparameters), (GDestroyNotify) ggml_hyperparameters_unref);
}

GGMLHyperparameters *
ggml_hyperparameters_load_from_istream_finish (GAsyncResult  *result,
                                               GError       **error)
{
  g_return_val_if_fail (g_task_is_valid (result, NULL), NULL);
  GTask *task = G_TASK (result);

  return g_task_propagate_pointer (task, error);
}

void
ggml_hyperparameters_load_from_istream_async (GInputStream *istream,
                                              GCancellable *cancellable,
                                              GAsyncReadyCallback callback,
                                              gpointer user_data)
{
  g_autoptr(GGMLHyperparametersLoadFromIstreamData) data = ggml_hyperparameters_load_from_istream_data_new(istream);

  g_autoptr(GTask) task = g_task_new (NULL, cancellable, callback, user_data);
  g_task_set_task_data (task, g_steal_pointer (&data), (GDestroyNotify) ggml_hyperparameters_load_from_istream_data_free);
  g_task_run_in_thread (task, ggml_hyperparameters_load_from_istream_async_thread);
}

/**
 * ggml_hyperparameters_get_int32:
 * @hyperparameters: A #GGMLHyperparameters
 * @key: The hyperparameter key to retrieve
 *
 * Gets the corresponding value for @key in the hyperparameters. It is an
 * error to pass an invalid key to this function.
 *
 * Returns: An #int32_t with the corresponding value
 */
int32_t
ggml_hyperparameters_get_int32 (GGMLHyperparameters *hyperparameters, const char *key)
{
  int32_t value;
  gboolean successful = g_hash_table_lookup_extended (hyperparameters->parameters, (gconstpointer) key, NULL, (gpointer *) &value);

  g_assert (successful == TRUE);

  return value;
}

/**
 * ggml_hyperparameters_ref:
 * @hyperparameters: A #GGMLHyperparameters
 *
 * Returns: (transfer full): The #GGMLHyperparameters with the increased ref count.
 */
GGMLHyperparameters *
ggml_hyperparameters_ref (GGMLHyperparameters *hyperparameters)
{
  ++hyperparameters->ref_count;
  return hyperparameters;
}

/**
 * ggml_hyperparameters_unref: (skip)
 * @hyperparameters: A #GGMLHyperparameters
 *
 * Frees the @hyperparameters
 */
void
ggml_hyperparameters_unref (GGMLHyperparameters *hyperparameters)
{
  if (--hyperparameters->ref_count == 0)
    {
      g_clear_pointer (&hyperparameters->parameters, g_hash_table_destroy);
      g_clear_pointer (&hyperparameters->ordered_keys, g_strfreev);
      g_clear_pointer (&hyperparameters, g_free);
    }
}

G_DEFINE_BOXED_TYPE (GGMLHyperparameters, ggml_hyperparameters, ggml_hyperparameters_ref, ggml_hyperparameters_unref)
