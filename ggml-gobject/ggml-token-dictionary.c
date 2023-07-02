/*
 * ggml-gobject/ggml-token-dictionary.c
 *
 * Library code for ggml-token-dictionary
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

#include <ggml-gobject/ggml-token-dictionary.h>
#include <ggml-gobject/internal/ggml-stream-internal.h>

struct _GGMLTokenDictionary {
  gchar **idx_to_word;
  GHashTable *word_to_idx;
  size_t ref_count;
};

/**
 * ggml_token_dictionary_new:
 * @tokens: (array zero-terminated=1): The tokens to add to this dictionary, in order
 *
 * Returns: (transfer full): A new #GGMLTokenDictionary
 */
GGMLTokenDictionary *
ggml_token_dictionary_new (const char **tokens)
{
  GGMLTokenDictionary *dictionary = g_new0 (GGMLTokenDictionary, 1);
  dictionary->idx_to_word = g_strdupv ((char **) tokens);
  dictionary->word_to_idx = g_hash_table_new_full (g_str_hash, g_str_equal, NULL, NULL);
  dictionary->ref_count = 1;

  int i = 0;
  const char **tokens_iterator = (const char **) dictionary->idx_to_word;

  while (*tokens_iterator != NULL)
    {
      g_hash_table_insert (dictionary->word_to_idx, (gpointer) *(tokens_iterator++), GINT_TO_POINTER (i++));
    }

  return dictionary;
}

/**
 * ggml_token_dictionary_load_from_istream:
 * @istream: (transfer none): A #GInputStream
 * @n_vocab: An #int32_t with the expected vocab size
 * @cancellable: (transfer none): A #GCancellable
 * @error: A #GError out variable
 *
 * Returns: (transfer full): A #GGMLTokenDictionary loaded from @istream or %NULL
 *          with @error set on failure.
 */
GGMLTokenDictionary *
ggml_token_dictionary_load_from_istream (GInputStream *istream,
                                         int32_t n_vocab,
                                         GCancellable *cancellable,
                                         GError **error)
{
  int32_t model_n_vocab_check;

  if (!ggml_input_stream_read_exactly (istream, (char *) &model_n_vocab_check, sizeof (int32_t) * 1, cancellable, error))
    {
      return FALSE;
    }

  if (model_n_vocab_check != n_vocab)
    {
      g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED, "Model dictionary n_vocab %d does not match hyperparameters n_vocab %d", model_n_vocab_check, n_vocab);
      return NULL;
    }

  g_autoptr (GPtrArray) words = g_ptr_array_new_full (n_vocab + 1, g_free);

  for (size_t i = 0; i < n_vocab; ++i)
    {
      uint32_t word_size;

      if (!ggml_input_stream_read_exactly (istream, (char *) &word_size, sizeof (uint32_t) * 1, cancellable, error))
        {
          return FALSE;
        }

      g_autofree char *buf = g_new0 (char, word_size + 1);

      if (!ggml_input_stream_read_exactly (istream, (char *) buf, sizeof (char) * word_size, cancellable, error))
        {
          return FALSE;
        }

      buf[word_size] = '\0';
      g_ptr_array_add (words, g_steal_pointer (&buf));
    }

  g_ptr_array_add (words, NULL);

  /* The strings will be copied into the dictionary and autofree'd from here */
  return ggml_token_dictionary_new ((const char **) words->pdata);
}

typedef struct _GGMLTokenDictionaryLoadFromIstreamData
{
  GInputStream *istream;
  int32_t n_vocab;
} GGMLTokenDictionaryLoadFromIstreamData;

static GGMLTokenDictionaryLoadFromIstreamData *
ggml_token_dictionary_load_from_istream_data_new (GInputStream *istream, int32_t n_vocab)
{
  GGMLTokenDictionaryLoadFromIstreamData *data = g_new0 (GGMLTokenDictionaryLoadFromIstreamData, 1);
  data->istream = g_object_ref (istream);
  data->n_vocab = n_vocab;

  return data;
}

static void
ggml_token_dictionary_load_from_istream_data_free (GGMLTokenDictionaryLoadFromIstreamData *data)
{
  g_clear_pointer (&data->istream, g_object_unref);
  g_clear_pointer (&data, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLTokenDictionaryLoadFromIstreamData, ggml_token_dictionary_load_from_istream_data_free);

static void
ggml_token_dictionary_load_from_istream_async_thread (GTask         *task,
                                                      gpointer       source_object,
                                                      gpointer       task_data,
                                                      GCancellable  *cancellable)
{
  GGMLTokenDictionaryLoadFromIstreamData *data = task_data;
  GError *error = NULL;

  g_autoptr(GGMLTokenDictionary) token_dictionary = ggml_token_dictionary_load_from_istream (data->istream,
                                                                                             data->n_vocab,
                                                                                             cancellable,
                                                                                             &error);

  if (token_dictionary == NULL)
    {
      g_task_return_error (task, error);
    }

  g_task_return_pointer (task, g_steal_pointer (&token_dictionary), (GDestroyNotify) ggml_token_dictionary_unref);
}

GGMLTokenDictionary *
ggml_token_dictionary_load_from_istream_finish (GAsyncResult  *result,
                                                GError       **error)
{
  g_return_val_if_fail (g_task_is_valid (result, NULL), NULL);
  GTask *task = G_TASK (result);

  return g_task_propagate_pointer (task, error);
}

void
ggml_token_dictionary_load_from_istream_async (GInputStream *istream,
                                               int32_t       n_vocab,
                                               GCancellable *cancellable,
                                               GAsyncReadyCallback callback,
                                               gpointer user_data)
{
  g_autoptr(GGMLTokenDictionaryLoadFromIstreamData) data = ggml_token_dictionary_load_from_istream_data_new (istream, n_vocab);

  g_autoptr(GTask) task = g_task_new (NULL, cancellable, callback, user_data);
  g_task_set_task_data (task, g_steal_pointer (&data), (GDestroyNotify) ggml_token_dictionary_load_from_istream_data_free);
  g_task_run_in_thread (task, ggml_token_dictionary_load_from_istream_async_thread);
}

/**
 * ggml_token_dictionary_ref: (skip)
 * @dictionary: A #GGMLTokenDictionary
 *
 * Returns: (transfer full): A #GGMLTokenDictionary
 */
GGMLTokenDictionary *
ggml_token_dictionary_ref (GGMLTokenDictionary *dictionary)
{
  ++dictionary->ref_count;
  return dictionary;
}

/**
 * ggml_token_dictionary_unref: (skip)
 * @dictionary: A #GGMLTokenDictionary
 *
 * Decrements the reference count on @dictionary and frees the underlying
 * dictionary if the reference count goes to zero.
 */
void
ggml_token_dictionary_unref (GGMLTokenDictionary *dictionary)
{
  if (--dictionary->ref_count == 0)
    {
      g_clear_pointer (&dictionary->word_to_idx, g_hash_table_destroy);
      g_clear_pointer (&dictionary->idx_to_word, g_strfreev);
      g_clear_pointer (&dictionary, g_free);
    }
}

/**
 * ggml_token_dictionary_lookup_extended:
 * @token_dictionary: A #GGMLTokenDictionary
 * @key: A key to look up in the @token_dictionary
 * @out_token: (out): The corresponding token
 *
 * Returns: %TRUE if the token was found in the dictionary and @value set
 *          %FALSE otherwise.
 */
gboolean
ggml_token_dictionary_lookup_extended (GGMLTokenDictionary *token_dictionary,
                                       const char *key,
                                       int32_t *out_token)
{
  gpointer lookup_token = NULL;

  if (g_hash_table_lookup_extended (token_dictionary->word_to_idx, key, NULL, (gpointer *) &lookup_token))
    {
      *out_token = GPOINTER_TO_INT (lookup_token);
      return TRUE;
    }

  return FALSE;
}

/**
 * ggml_token_dictionary_decode:
 * @token_dictionary: (transfer none): A #GGMLTokenDictionary
 * @tokens: (array length=n_tokens): An array of #int32_t tokens
 * @n_tokens: Number of tokens in @tokens
 *
 * Decode the token array back into a string. It is an error to
 * pass tokens to this function which are outside the range of tokens
 * in @token_dictionary.
 *
 * Returns: (transfer full) (array zero-terminated=1): A new string with the decoded tokens.
 */
char *
ggml_token_dictionary_decode (GGMLTokenDictionary *token_dictionary,
                              int32_t             *tokens,
                              size_t               n_tokens)
{
  size_t token_dictionary_size = g_hash_table_size (token_dictionary->word_to_idx);
  g_autoptr(GPtrArray) decoded_tokens = g_ptr_array_new_null_terminated (n_tokens, NULL, TRUE);

  for (size_t i = 0; i < n_tokens; ++i)
    {
      g_assert (tokens[i] < token_dictionary_size);

      g_ptr_array_add (decoded_tokens, token_dictionary->idx_to_word[tokens[i]]);
    }

  return g_strjoinv ("", (char **) decoded_tokens->pdata);
}

G_DEFINE_BOXED_TYPE (GGMLTokenDictionary, ggml_token_dictionary, ggml_token_dictionary_ref, ggml_token_dictionary_unref)
