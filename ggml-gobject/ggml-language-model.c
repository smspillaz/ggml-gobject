/*
 * ggml-gobject/ggml-language-model.c
 *
 * Library code for ggml-language-model
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

#include <ggml-gobject/ggml-gpt.h>
#include <ggml-gobject/ggml-language-model.h>
#include <ggml-gobject/internal/ggml-stream-internal.h>

struct _GGMLLanguageModel {
  GGMLHyperparameters *hyperparameters;
  GGMLTokenDictionary *token_dictionary;
  GGMLModel *model;
  size_t ref_count;
};

#define GGML_LANGUAGE_MODEL_MAGIC 0x67676d6c

/**
 * ggml_language_model_consume_istream_magic:
 * @istream: A #GInputStream
 * @cancellable: A #GCancellable
 * @error: A #GError
 *
 * Returns: %TRUE if the operation succeeded, %FALSE with @error set on failure.
 */
gboolean
ggml_language_model_consume_istream_magic (GInputStream *istream,
                                           GCancellable *cancellable,
                                           GError **error)
{
  uint32_t magic;

  if (!ggml_input_stream_read_exactly (istream, (char *) &magic, sizeof (uint32_t), cancellable, error))
    return FALSE;

  if (magic != GGML_LANGUAGE_MODEL_MAGIC)
    {
      g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED, "Invalid magic %#010x expected %#010x", magic, GGML_LANGUAGE_MODEL_MAGIC);
      return FALSE;
    }

  return TRUE;
}

static void
ggml_language_model_consume_istream_magic_thread (GTask *task,
                                                  gpointer source_object,
                                                  gpointer user_data,
                                                  GCancellable *cancellable)
{
  GInputStream *istream = user_data;
  GError *error = NULL;

  if (!ggml_language_model_consume_istream_magic (istream, cancellable, &error))
    {
      g_task_return_error (task, error);
      return;
    }

  g_task_return_boolean (task, TRUE);
}

gboolean
ggml_language_model_consume_istream_magic_finish (GAsyncResult  *result,
                                                  GError      **error)
{
  g_return_val_if_fail (g_task_is_valid (result, NULL), FALSE);

  GTask *task = G_TASK (result);

  return g_task_propagate_boolean (task, error);
}

/**
 * ggml_language_model_consume_istream_magic_async:
 * @istream: A #GInputStream
 * @cancellable: A #GCancellable
 * @callback: A #GAsyncReadyCallback
 * @user_data: (closure callback): A gpointer to some data for @callback
 */
void
ggml_language_model_consume_istream_magic_async (GInputStream         *istream,
                                                 GCancellable         *cancellable,
                                                 GAsyncReadyCallback   callback,
                                                 gpointer              user_data)
{
  g_autoptr(GTask) task = g_task_new (NULL, cancellable, callback, user_data);
  g_task_set_task_data (task, g_object_ref (istream), g_object_unref);
  g_task_run_in_thread (task, ggml_language_model_consume_istream_magic_thread);
}

/**
 * ggml_language_model_new:
 * @hyperparameters: A #GGMLHyperparameters
 * @dictionary: A #GGMLTokenDictionary
 * @model: A #GGMLModel
 *
 * Creates a new #GGMLLanguageModel
 *
 * Returns: (transfer full): A new #GGMLLanguageModel
 */
GGMLLanguageModel *
ggml_language_model_new (GGMLHyperparameters *hyperparameters, GGMLTokenDictionary *dictionary, GGMLModel *model)
{
  GGMLLanguageModel *language_model = g_new0 (GGMLLanguageModel, 1);
  language_model->hyperparameters = ggml_hyperparameters_ref (hyperparameters);
  language_model->token_dictionary = ggml_token_dictionary_ref (dictionary);
  language_model->model = ggml_model_ref (model);
  language_model->ref_count = 1;

  return language_model;
}

static size_t
argmax_f (float *elements, size_t num_elements)
{
  size_t max_idx = 0;
  float max_val = -G_MAXFLOAT;

  for (size_t i = 0; i < num_elements; ++i)
    {
      if (elements[i] > max_val)
        {
          max_idx = i;
          max_val = elements[i];
        }
    }

  return max_idx;
}

static gboolean
ggml_language_model_forward_single_iteration (GGMLModel            *model,
                                              GGMLHyperparameters  *hyperparameters,
                                              GHashTable           *inference_parameters,
                                              GBytes               *mem_buffer,
                                              int32_t              *input_tokens,
                                              size_t                n_input_tokens,
                                              int32_t              *out_token,
                                              GError              **error)
{
  int32_t n_vocab = ggml_hyperparameters_get_int32 (hyperparameters, "n_vocab");
  g_autoptr(GVariant) variant = g_variant_ref_sink(g_variant_new_fixed_array (G_VARIANT_TYPE_INT32,
                                                                              input_tokens,
                                                                              n_input_tokens,
                                                                              sizeof (int32_t)));
  g_autoptr(GGMLTensor) logits_tensor = ggml_model_forward (model,
                                                            hyperparameters,
                                                            variant,
                                                            inference_parameters,
                                                            mem_buffer,
                                                            error);

  if (logits_tensor == NULL)
    {
      *out_token = -1;
      return FALSE;
    }

  size_t logits_tensor_n_bytes;
  float *logits_tensor_data = (float *) ggml_tensor_get_data (logits_tensor, &logits_tensor_n_bytes);
  float *end_logit_data = logits_tensor_data + (((int32_t) (n_input_tokens - 1)) * n_vocab);
  *out_token = argmax_f (end_logit_data, n_vocab);

  return TRUE;
}

static const char n_past_key[] = "n_past";

static int32_t *
ggml_language_model_forward_loop (GGMLModel           *model,
                                  GGMLHyperparameters *hyperparameters,
                                  int32_t             *initial_prompt_tokens,
                                  size_t               n_initial_prompt_tokens,
                                  int32_t              num_iterations,
                                  size_t              *out_num_tokens,
                                  GError             **error)
{
  /* Assming for now that num_iterations is positive */
  g_autoptr(GArray) prompt_tokens = g_array_sized_new (FALSE, TRUE, sizeof (int32_t), n_initial_prompt_tokens + num_iterations);
  g_autoptr(GHashTable) inference_parameters = g_hash_table_new_full (g_str_hash, g_str_equal, NULL , NULL);

  g_hash_table_insert (inference_parameters, (gpointer) n_past_key, GINT_TO_POINTER (0));

  memcpy (prompt_tokens->data, initial_prompt_tokens, sizeof (int32_t) * n_initial_prompt_tokens);
  prompt_tokens->len = n_initial_prompt_tokens;

  /* Edge case */
  if (num_iterations == 0)
    {
      return g_array_steal (prompt_tokens, out_num_tokens);
    }

  /* Create a memory buffer for prompt_tokens->len. Because we do subsequent
   * passes using a single query (saved keys and values), we only need to allocate
   * this much memory. */
  g_autoptr(GBytes) mem_buffer = ggml_gpt_model_forward_pass_create_memory_buffer (n_initial_prompt_tokens + num_iterations);

  /* We first do a single iteration to populate the key/value memories */
  int32_t argmax;
  if (!ggml_language_model_forward_single_iteration (model,
                                                     hyperparameters,
                                                     inference_parameters,
                                                     mem_buffer,
                                                     (int32_t *) prompt_tokens->data,
                                                     prompt_tokens->len,
                                                     &argmax,
                                                     error))
    {
      return NULL;
    }

  g_array_append_vals (prompt_tokens, &argmax, 1);

  /* Now we have the key/value memories and we can do the inference as usual.
   *
   * Here we pass in one token at a time, eg, the length of the input is always 1
   * and we are using the most recent token. The keys/values from previous iterations
   * are cached. This means that decoding performance can be linear, as opposed
   * to quadratic. */
  for (int32_t i = 0; i < num_iterations - 1; ++i)
    {
      g_hash_table_insert (inference_parameters, (gpointer) n_past_key, GINT_TO_POINTER (n_initial_prompt_tokens + i));

      if (!ggml_language_model_forward_single_iteration (model,
                                                         hyperparameters,
                                                         inference_parameters,
                                                         mem_buffer,
                                                         ((int32_t *) prompt_tokens->data) + n_initial_prompt_tokens + i,
                                                         1,
                                                         &argmax,
                                                         error))
        {
          return NULL;
        }

      g_array_append_vals (prompt_tokens, &argmax, 1);
    }

  *out_num_tokens = 0;

  return g_array_steal (prompt_tokens, out_num_tokens);
}

/**
 * ggml_language_model_decode_tokens:
 * @language_model: A #GGMLLanguageModel
 * @tokens: (array length=length): An #int32_t array of tokens
 * @length: The length of @tokens
 *
 * Returns: (transfer full): The decoded tokens
 */
char *
ggml_language_model_decode_tokens (GGMLLanguageModel *language_model,
                                   int32_t           *tokens,
                                   size_t             length)
{
  return ggml_token_dictionary_decode (language_model->token_dictionary,
                                       tokens,
                                       length);
}

/**
 * ggml_language_model_complete:
 * @language_model: A #GGMLLanguageModel
 * @prompt: An input prompt
 * @num_iterations: Number of tokens to generate.
 * @out_is_complete_eos: (out): An out-variable indicating whether we hit an EOS token.
 * @error: A #GError
 *
 * Returns: (transfer full): The completed prompt, after running the autoregressive
 *          generation procedure for @num_iterations.
 */
char *
ggml_language_model_complete (GGMLLanguageModel  *language_model,
                              const char         *prompt,
                              int32_t             num_iterations,
                              gboolean           *out_is_complete_eos,
                              GError            **error)
{
  g_autofree int32_t *tokens = NULL;
  size_t   n_tokens = 0;

  if (!ggml_gpt_tokenize (language_model->token_dictionary, prompt, &tokens, &n_tokens, error))
    {
      return NULL;
    }

  size_t out_num_tokens = 0;
  g_autofree int32_t *completed_tokens = ggml_language_model_forward_loop (language_model->model,
                                                                           language_model->hyperparameters,
                                                                           tokens,
                                                                           n_tokens,
                                                                           num_iterations,
                                                                           &out_num_tokens,
                                                                           error);

  if (completed_tokens == NULL)
    {
      return NULL;
    }

  /* May be used in the future, but for now always %FALSE */
  *out_is_complete_eos = FALSE;
  return ggml_token_dictionary_decode (language_model->token_dictionary,
                                       completed_tokens,
                                       out_num_tokens);
}

typedef struct _GGMLLanguageModelChunkCompletionResult
{
  char     *chunk;
  gboolean  is_complete;
  gboolean  is_complete_eos;
} GGMLLanguageModelChunkCompletionResult;

typedef struct _GGMLLanguageModelChunkCompletion
{
  GGMLLanguageModelChunkCompletionResult *result;
  GError *error;
} GGMLLanguageModelChunkCompletion;

/**
 * ggml_language_model_chunk_completion_result_new:
 * @chunk: (transfer full): A string with the chunk data
 * @is_complete: A boolean value indicating whether decoding is complete.
 * @is_complete_eos: A boolean value indicating whether decoding was complete
 *                   because we hit an EOS value.
 *
 * Returns: (transfer full): A new #GGMLLanguageModelChunkCompletionResult
 */
static GGMLLanguageModelChunkCompletionResult *
ggml_language_model_chunk_completion_result_new (char     *chunk,
                                                 gboolean  is_complete,
                                                 gboolean  is_complete_eos)
{
  GGMLLanguageModelChunkCompletionResult *result = g_new0 (GGMLLanguageModelChunkCompletionResult, 1);
  result->chunk = g_steal_pointer (&chunk);
  result->is_complete = is_complete;
  result->is_complete_eos = is_complete_eos;

  return result;
}

static void
ggml_language_model_chunk_completion_result_free (GGMLLanguageModelChunkCompletionResult *result)
{
  g_clear_pointer (&result->chunk, g_free);
  g_clear_pointer (&result, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLLanguageModelChunkCompletionResult, ggml_language_model_chunk_completion_result_free)

/**
 * ggml_language_model_chunk_completion_new:
 * @result: (transfer full): A #GGMLLanguageModelChunkCompletionResult
 * @error: (transfer full) (nullable): A GError or %NULL
 *
 * Returns: (transfer full): A new #GGMLLanguageModelChunkCompletion
 */
static GGMLLanguageModelChunkCompletion *
ggml_language_model_chunk_completion_new (GGMLLanguageModelChunkCompletionResult  *result,
                                          GError                                  *error)
{
  GGMLLanguageModelChunkCompletion *chunk_completion = g_new0 (GGMLLanguageModelChunkCompletion, 1);
  chunk_completion->result = g_steal_pointer (&result);

  if (error != NULL)
    {
      g_propagate_error (&chunk_completion->error, error);
    }

  return chunk_completion;
}

static void
ggml_language_model_chunk_completion_free (GGMLLanguageModelChunkCompletion *completion)
{
  g_clear_pointer (&completion->result, ggml_language_model_chunk_completion_result_free);
  g_clear_pointer (&completion->error, g_error_free);

  g_clear_pointer (&completion, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLLanguageModelChunkCompletion, ggml_language_model_chunk_completion_free)


/**
 * ggml_language_model_complete_finish:
 * @language_model: A #GGMLLanguageModel
 * @result: A #GAsyncResult
 * @out_is_complete: (out): An output variable indicating if the number of
 *                   tokens requested has been generated or the completion has
 *                   reached an end-of-sentence token.
 * @out_is_complete_eos: (out): An output variable indicating if the
 *                              completion has reached an end-of-sentence token.
 * @error: A #GError
 *
 * Complete the call to ggml_language_model_complete_async and return
 * a pointer to an char * array of a string chunk (of number of tokens chunk_size
 * given to ggml_language_model_complete_async).
 *
 * There are two "complete" states. The first, @out_is_complete, just means
 * that we've generated at least as much as has been requested, but not necessarily
 * that the sequence could be further completed (eg, by passing a new prompt with
 * the same sequence). The second, @out_is_complete_eos means the sequence has
 * been completed to an end-of-sequence token, so the model had nothing left
 * to generate for this sequence and finished early.
 *
 * Returns: (transfer full): A new string or %NULL with @error set on
 *          failure.
 */
char *
ggml_language_model_complete_finish (GGMLLanguageModel  *language_model,
                                     GAsyncResult       *result,
                                     gboolean           *out_is_complete,
                                     gboolean           *out_is_complete_eos,
                                     GError            **error)
{
  GTask *task = G_TASK (result);
  g_autoptr(GGMLLanguageModelChunkCompletionResult) results = g_task_propagate_pointer (task, error);

  if (results == NULL)
    {
      return NULL;
    }

  *out_is_complete = results->is_complete;
  *out_is_complete_eos = results->is_complete_eos;

  return g_strdup (results->chunk);
}

typedef struct _GGMLLanguageModelCompleteState
{
  GGMLLanguageModel *language_model;
  char *prompt;
  GArray *prompt_tokens;
  size_t iterations;
  size_t chunk_size;
  GAsyncQueue *queue;
} GGMLLanguageModelCompleteState;

static GGMLLanguageModelCompleteState *
ggml_language_model_complete_state_new (GGMLLanguageModel    *language_model,
                                        const char           *prompt,
                                        size_t                iterations,
                                        size_t                chunk_size,
                                        GAsyncQueue          *async_queue)
{
  GGMLLanguageModelCompleteState *state = g_new0 (GGMLLanguageModelCompleteState, 1);
  state->language_model = ggml_language_model_ref (language_model);
  state->prompt = g_strdup (prompt);
  state->iterations = iterations;
  state->chunk_size = chunk_size;
  state->queue = g_async_queue_ref (async_queue);

  return state;
}

static void
ggml_language_model_complete_state_free (GGMLLanguageModelCompleteState *state)
{
  g_clear_pointer (&state->queue, g_async_queue_unref);
  g_clear_pointer (&state->prompt, g_free);

  g_clear_pointer (&state, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLLanguageModelCompleteState, ggml_language_model_complete_state_free)

static void
ggml_language_model_complete_thread_queue_push (GGMLLanguageModelCompleteState *state,
                                                GGMLLanguageModelChunkCompletion *completion)
{
  /* We assume that this succeeds */
  g_async_queue_push (state->queue, completion);
  g_main_context_wakeup (g_main_context_default ());
}

static void
ggml_language_model_complete_thread_push_tokens_or_error (GGMLLanguageModelCompleteState *state,
                                                          char      *chunk,
                                                          gboolean  is_complete,
                                                          gboolean  is_complete_eos,
                                                          GError   *error)
{
  if (error != NULL)
    {
      g_autoptr(GGMLLanguageModelChunkCompletion) completion = ggml_language_model_chunk_completion_new (
        NULL,
        g_steal_pointer (&error)
      );
      ggml_language_model_complete_thread_queue_push (state, g_steal_pointer (&completion));
      return;
    }

  g_autoptr(GGMLLanguageModelChunkCompletionResult) result = ggml_language_model_chunk_completion_result_new(
    chunk,
    is_complete,
    is_complete_eos
  );
  g_autoptr(GGMLLanguageModelChunkCompletion) completion = ggml_language_model_chunk_completion_new (
    g_steal_pointer (&result),
    NULL
  );
  ggml_language_model_complete_thread_queue_push (state, g_steal_pointer (&completion));
}

static gpointer
ggml_language_model_complete_thread_loop (gpointer data)
{
  GGMLLanguageModelCompleteState *state = data;
  g_autofree int32_t *out_prompt_tokens = NULL;
  size_t   out_n_prompt_tokens = 0;
  int32_t  n_completed_iterations = 0;
  g_autoptr(GError) error = NULL;

  if (!ggml_gpt_tokenize (state->language_model->token_dictionary,
                          state->prompt,
                          &out_prompt_tokens,
                          &out_n_prompt_tokens,
                          &error))
    {
      ggml_language_model_complete_thread_push_tokens_or_error (state,
                                                                NULL,
                                                                FALSE,
                                                                FALSE,
                                                                g_steal_pointer (&error));
      return GINT_TO_POINTER (FALSE);
    }

  /* Immediately return this chunk back to the caller. They will need to
   * collect the tokens. */
  g_autofree char *init_chunk = g_strdup(state->prompt);
  /* We completed a chunk, send it to the caller. */
  ggml_language_model_complete_thread_push_tokens_or_error (state,
                                                            g_steal_pointer (&init_chunk),
                                                            FALSE,
                                                            FALSE,
                                                            NULL);

  /* Create a memory buffer for prompt_tokens->len. Because we do subsequent
   * passes using a single query (saved keys and values), we only need to allocate
   * this much memory. */
  g_autoptr(GBytes) mem_buffer = ggml_gpt_model_forward_pass_create_memory_buffer (out_n_prompt_tokens + state->iterations);
  g_autoptr(GHashTable) inference_parameters = g_hash_table_new_full (g_str_hash, g_str_equal, NULL , NULL);

  g_hash_table_insert (inference_parameters, (gpointer) n_past_key, GINT_TO_POINTER (0));

  /* We first do a single iteration to populate the key/value memories */
  int32_t current_chunk_index = 0;
  g_autoptr(GArray) chunk_tokens = g_array_sized_new (FALSE, TRUE, sizeof (int32_t), state->chunk_size);
  int32_t *chunk_tokens_data = &(g_array_index (chunk_tokens, int32_t, 0));

  /* We allocate enough space upfront and also set the size.
   * This is used as a ring buffer, so all zeros are fine */
  chunk_tokens->len = state->chunk_size;

  if (!ggml_language_model_forward_single_iteration (state->language_model->model,
                                                     state->language_model->hyperparameters,
                                                     inference_parameters,
                                                     mem_buffer,
                                                     out_prompt_tokens,
                                                     out_n_prompt_tokens,
                                                     &chunk_tokens_data[current_chunk_index],
                                                     &error))
    {
      ggml_language_model_complete_thread_push_tokens_or_error (state,
                                                                NULL,
                                                                FALSE,
                                                                FALSE,
                                                                g_steal_pointer (&error));
      return GINT_TO_POINTER (FALSE);
    }

  ++n_completed_iterations;

  for (; n_completed_iterations < state->iterations; ++n_completed_iterations)
    {
      current_chunk_index = n_completed_iterations % state->chunk_size;
      int32_t read_chunk_index = (current_chunk_index - 1 + state->chunk_size) % state->chunk_size;
      g_assert (read_chunk_index >= 0);

      g_hash_table_insert (inference_parameters,
                           (gpointer) n_past_key,
                           GINT_TO_POINTER (out_n_prompt_tokens + n_completed_iterations - 1));

      if (current_chunk_index == 0)
        {
          char *chunk = ggml_token_dictionary_decode (state->language_model->token_dictionary,
                                                      chunk_tokens_data,
                                                      chunk_tokens->len);
          /* We completed a chunk, send it to the caller. */
          ggml_language_model_complete_thread_push_tokens_or_error (state,
                                                                    g_steal_pointer (&chunk),
                                                                    FALSE,
                                                                    FALSE,
                                                                    NULL);
        }

      if (!ggml_language_model_forward_single_iteration (state->language_model->model,
                                                         state->language_model->hyperparameters,
                                                         inference_parameters,
                                                         mem_buffer,
                                                         &chunk_tokens_data[read_chunk_index],
                                                         1,
                                                         &chunk_tokens_data[current_chunk_index],
                                                         &error))
        {
          ggml_language_model_complete_thread_push_tokens_or_error (state,
                                                                    NULL,
                                                                    FALSE,
                                                                    FALSE,
                                                                    g_steal_pointer (&error));
          return GINT_TO_POINTER (FALSE);
        }
    }

    g_array_set_size (chunk_tokens, (n_completed_iterations - 1) % state->chunk_size + 1);
    g_autofree char *chunk = ggml_token_dictionary_decode (state->language_model->token_dictionary,
                                                           (int32_t *) chunk_tokens->data,
                                                           chunk_tokens->len);

    /* We completed a chunk, send it to the caller. */
    ggml_language_model_complete_thread_push_tokens_or_error (state,
                                                              g_steal_pointer (&chunk),
                                                              TRUE,
                                                              FALSE,
                                                              NULL);

  return GINT_TO_POINTER (TRUE);
}

typedef struct _GGMLLanguageModelCompleteMonitorState
{
  GAsyncQueue *queue;
  GAsyncReadyCallback callback;
  gpointer user_data;
  GDestroyNotify user_data_destroy;
} GGMLLanguageModelCompleteMonitorState;

static GGMLLanguageModelCompleteMonitorState *
ggml_language_model_complete_monitor_state_new (GAsyncQueue          *async_queue,
                                                GAsyncReadyCallback   callback,
                                                gpointer              user_data,
                                                GDestroyNotify        user_data_destroy)
{
  GGMLLanguageModelCompleteMonitorState *state = g_new0 (GGMLLanguageModelCompleteMonitorState, 1);
  state->callback = callback;
  state->user_data = user_data;
  state->user_data_destroy = user_data_destroy;
  state->queue = g_async_queue_ref (async_queue);

  return state;
}

static void
ggml_language_model_complete_monitor_state_free (GGMLLanguageModelCompleteMonitorState *state)
{
  g_clear_pointer (&state->queue, g_async_queue_unref);
  g_clear_pointer (&state->user_data, state->user_data_destroy);

  g_clear_pointer (&state, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLLanguageModelCompleteMonitorState, ggml_language_model_complete_monitor_state_free)

/**
 * ggml_language_model_monitor_process_completion: (skip)
 * @completion: A #GGMLLanguageModelChunkCompletion fragment
 * @state: A #GGMLLanguageModelCompleteMonitorState
 *
 * Processes a partial completion and returns either an error
 * or a partial completion to the GAsyncReadyCallback.
 *
 * Returns: %TRUE if the monitor source should be terminated (eg, the
 *          thread has reached some end condition) or %FALSE
 *          if the monitor source should continue.
 */
static gboolean
ggml_language_model_monitor_process_completion (GGMLLanguageModelCompleteMonitorState *state,
                                                GGMLLanguageModelChunkCompletion      *completion)
{
  g_autoptr(GTask) task = g_task_new (NULL, NULL, state->callback, state->user_data);

  if (completion->error != NULL)
    {
      GError *error = NULL;
      g_propagate_error (&error, g_steal_pointer (&completion->error));
      g_task_return_error (task, error);
      return TRUE;
    }

  if (completion->result != NULL)
    {
      g_task_return_pointer (task, g_steal_pointer (&completion->result), NULL);
      return FALSE;
    }

  /* We're all done, return TRUE because we got a sentinel value */
  return TRUE;
}

static gboolean
ggml_language_model_monitor_callback (gpointer user_data)
{
  GGMLLanguageModelCompleteMonitorState *state = user_data;

  while (TRUE)
    {
      GGMLLanguageModelChunkCompletion *completion = g_async_queue_try_pop (state->queue);

      /* Nothing more in the queue, but we're not done with our completion,
       * so return G_SOURCE_CONTINUE here. */
      if (completion == NULL)
        {
          return G_SOURCE_CONTINUE;
        }

      /* If we return TRUE*/
      if (ggml_language_model_monitor_process_completion (state, completion))
        {
          ggml_language_model_chunk_completion_free (completion);
          ggml_language_model_complete_monitor_state_free (state);
          return G_SOURCE_REMOVE;
        }

      ggml_language_model_chunk_completion_free (completion);
    }
}

/**
 * ggml_language_model_complete_async:
 * @language_model: (transfer none): A #GGMLLanguageModel
 * @prompt: (transfer none): An initial prompt to use for the model
 * @num_iterations: Number of additional tokens to generate
 * @chunk_size: Chunk size of tokens that get sent to @callback on generation
 * @cancellable: (transfer none) (nullable): A #GCancellable
 * @callback: A #GAsyncReadyCallback to send partially completed sequences too
 * @user_data: (closure callback): Some user data for @callback
 * @user_data_destroy: (destroy callback): A destroy function for @user_data
 *
 * Asynchronously complete a prompt from @language_model by generating @num_iterations tokens.
 * This function can be used for streaming the result of generation, if @chunk_size < @num_iterations
 * as @callback will be invoked multiple times. @chunk_size is something that will need to be tuned according
 * to the needs of the application. A smaller @chunk_size means lower latency but higher overhead and
 * therefore slower overall generation.
 *
 * The @callback should not free its @user_data on
 * invocation - the generation process will do that once generation is complete. You can complete
 * a call to this function with ggml_language_model_complete_finish.
 *
 * Returns: (transfer full): A new #GThread
 */
GThread *
ggml_language_model_complete_async (GGMLLanguageModel    *language_model,
                                    const char           *prompt,
                                    size_t                num_iterations,
                                    size_t                chunk_size,
                                    GCancellable         *cancellable,
                                    GAsyncReadyCallback   callback,
                                    gpointer              user_data,
                                    GDestroyNotify        user_data_destroy)
{
  g_autoptr(GTask) task = g_task_new (NULL, cancellable, callback, user_data);
  g_autoptr(GError) error = NULL;

  g_autoptr(GAsyncQueue) async_queue = g_async_queue_new_full ((GDestroyNotify) ggml_language_model_chunk_completion_free);
  g_autoptr(GGMLLanguageModelCompleteMonitorState) monitor_state = ggml_language_model_complete_monitor_state_new (async_queue,
                                                                                                                   callback,
                                                                                                                   user_data,
                                                                                                                   user_data_destroy);

  g_idle_add ((GSourceFunc) ggml_language_model_monitor_callback,
              g_steal_pointer (&monitor_state));

  GGMLLanguageModelCompleteState *state = ggml_language_model_complete_state_new (language_model,
                                                                                  prompt,
                                                                                  num_iterations,
                                                                                  chunk_size,
                                                                                  async_queue);

  return g_thread_new ("complete-thread", ggml_language_model_complete_thread_loop, state);
}

static void
ggml_model_set_possible_tied_weights (GGMLModel *model,
                                      const char **loaded_keys,
                                      const char **src_weights,
                                      const char **dst_weights)
{
  const char **src_weights_it = src_weights;
  const char **dst_weights_it = dst_weights;

  for (; *src_weights_it != NULL && *dst_weights_it != NULL; ++src_weights_it, ++dst_weights_it)
    {
      if (!g_strv_contains (loaded_keys, *dst_weights_it) && g_strv_contains (loaded_keys, *src_weights_it))
        {
          GGMLTensor *src_tensor = ggml_model_get (model, *src_weights_it);
          GGMLTensor *dst_tensor = ggml_model_get (model, *dst_weights_it);

          g_assert (src_tensor != NULL);
          g_assert (dst_tensor != NULL);

          size_t src_n_bytes = 0;
          char *src_data = ggml_tensor_get_data (src_tensor, &src_n_bytes);
          ggml_tensor_set_data (dst_tensor, src_data, src_n_bytes);
        }
    }
}

/**
 * ggml_language_model_load_from_istream:
 * @istream: (transfer none): A #GInputStream
 * @create_model_desc: (transfer none) (scope call): A #GGMLModelDescFromHyperparametersFunc to specify the model structure and weights
 * @create_model_desc_user_data: (closure create_model_desc): A closure for @create_model_desc
 * @forward_func: (scope notified) (nullable): A #GGMLModelFowardFunc
 * @forward_func_user_data: (closure forward_func) (transfer full): The user data for @forward_func
 * @forward_func_user_data_destroy: (destroy forward_func): A #GDestroyNotify for forward_func
 * @cancellable: (transfer none) (nullable): A #GCancellable
 * @error: (nullable): A #GError
 *
 * Reads a GGML language model from @istream , which includes the hyperparameters, token
 * dictionary and model weights and returns a #GGMLLanguageModel
 *
 * Returns: (transfer full): A #GGMLLanguageModel with the loaded weights on success
 */
GGMLLanguageModel *
ggml_language_model_load_from_istream (GInputStream *istream,
                                       GGMLModelDescFromHyperparametersFunc create_model_desc,
                                       gpointer create_model_desc_user_data,
                                       GGMLModelForwardFunc forward_func,
                                       gpointer forward_func_user_data,
                                       GDestroyNotify forward_func_user_data_destroy,
                                       GCancellable *cancellable,
                                       GError **error)
{
  if (!ggml_language_model_consume_istream_magic (istream, cancellable, error))
    {
      return NULL;
    }

  g_autoptr(GGMLHyperparameters) hyperparameters = ggml_hyperparameters_load_from_istream (istream, cancellable, error);

  if (hyperparameters == NULL)
    {
      return NULL;
    }

  g_autoptr (GGMLModelDescNode) model_desc_node = (*create_model_desc) (hyperparameters, create_model_desc_user_data);
  int32_t n_vocab = ggml_hyperparameters_get_int32 (hyperparameters, "n_vocab");
  g_autoptr(GGMLTokenDictionary) token_dictionary = ggml_token_dictionary_load_from_istream (istream,
                                                                                             n_vocab,
                                                                                             cancellable,
                                                                                             error);

  if (token_dictionary == NULL)
    {
      return NULL;
    }

  g_auto(GStrv) loaded_keys = NULL;
  g_autoptr(GGMLModel) model = ggml_model_load_from_istream (istream,
                                                             model_desc_node,
                                                             hyperparameters,
                                                             forward_func,
                                                             forward_func_user_data,
                                                             forward_func_user_data_destroy,
                                                             &loaded_keys,
                                                             cancellable,
                                                             error);

  if (model == NULL)
    {
      return NULL;
    }

  const char *src_weights[] = {"model/wte", NULL};
  const char *dst_weights[] = {"model/lm_head", NULL};
  ggml_model_set_possible_tied_weights (model, (const char **) loaded_keys, src_weights, dst_weights);

  return ggml_language_model_new (hyperparameters,
                                  token_dictionary,
                                  model);
}

static struct GGMLLanguageModelDefinitions {
  GGMLModelDescFromHyperparametersFunc model_desc_from_hyperparameters_func;
  GGMLModelForwardFunc forward_func;
} ggml_language_model_definitions[] = {
  /* GGML_DEFINED_MODEL_GPT2 */
  {
    .model_desc_from_hyperparameters_func = (GGMLModelDescFromHyperparametersFunc) ggml_create_gpt2_model_desc_from_hyperparameters,
    .forward_func = ggml_gpt_model_forward_pass
  }
};

/**
 * ggml_language_model_load_defined_from_istream:
 * @model: A #GGMLDefinedLanguageModel configuration to load
 * @istream: (transfer none): A #GInputStream
 * @cancellable: (transfer none): A #GCancellable
 * @error: A #GError
 *
 * Load a GGMLLanguageModel according to some preset given by @model. This
 * is more language binding friendly, because it doesn't require calling
 * back into the bindings for finalize the model configuration, though it is
 * a litle more inflexible.
 *
 * Returns: (transfer full): A #GGMLLanguageModel on success or %NULL with
 *          @error set on failure
 */
GGMLLanguageModel *
ggml_language_model_load_defined_from_istream (GGMLDefinedLanguageModel    model,
                                               GInputStream               *istream,
                                               GCancellable               *cancellable,
                                               GError                    **error)
{
  return ggml_language_model_load_from_istream (istream,
                                                ggml_language_model_definitions[model].model_desc_from_hyperparameters_func,
                                                NULL,
                                                ggml_language_model_definitions[model].forward_func,
                                                NULL,
                                                NULL,
                                                cancellable,
                                                error);
}

typedef struct _GGMLLanguageModelLoadFromIstreamData
{
  GInputStream *istream;
  GGMLModelDescFromHyperparametersFunc create_model_desc;
  gpointer create_model_desc_user_data;
  GDestroyNotify create_model_desc_user_data_destroy;
  GGMLModelForwardFunc forward_func;
  gpointer forward_func_user_data;
  GDestroyNotify forward_func_user_data_destroy;

  /* Things that get loaded as we go */
  GGMLModelDescNode *model_desc;
  GGMLHyperparameters *hyperparameters;
  GGMLTokenDictionary *token_dictionary;
  GGMLModel *model;
} GGMLLanguageModelLoadFromIstreamData;

static GGMLLanguageModelLoadFromIstreamData *
ggml_language_model_load_from_istream_data_new (GInputStream *istream,
                                                GGMLModelDescFromHyperparametersFunc create_model_desc,
                                                gpointer create_model_desc_user_data,
                                                GDestroyNotify create_model_desc_user_data_destroy,
                                                GGMLModelForwardFunc forward_func,
                                                gpointer forward_func_user_data,
                                                GDestroyNotify forward_func_user_data_destroy)
{
  GGMLLanguageModelLoadFromIstreamData *data = g_new0 (GGMLLanguageModelLoadFromIstreamData, 1);

  data->istream = g_object_ref (istream);
  data->create_model_desc = create_model_desc;
  data->create_model_desc_user_data = create_model_desc_user_data;
  data->create_model_desc_user_data_destroy = create_model_desc_user_data_destroy;
  data->forward_func = forward_func;
  data->forward_func_user_data = forward_func_user_data;
  data->forward_func_user_data_destroy = forward_func_user_data_destroy;

  return data;
}

void
ggml_language_model_load_from_istream_data_free (GGMLLanguageModelLoadFromIstreamData *data)
{
  g_clear_pointer (&data->istream, g_object_unref);
  g_clear_pointer (&data->create_model_desc_user_data, data->create_model_desc_user_data_destroy);
  g_clear_pointer (&data->forward_func_user_data, data->forward_func_user_data_destroy);

  g_clear_pointer (&data->model_desc, ggml_model_desc_node_unref);
  g_clear_pointer (&data->model, ggml_model_unref);
  g_clear_pointer (&data->hyperparameters, ggml_hyperparameters_unref);
  g_clear_pointer (&data->token_dictionary, ggml_token_dictionary_unref);

  g_free (data);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLLanguageModelLoadFromIstreamData, ggml_language_model_load_from_istream_data_free)

static void
ggml_language_model_load_from_istream_on_model_read (GObject *src,
                                                     GAsyncResult *result,
                                                     gpointer user_data)
{
  GError *error = NULL;

  /* We take ownership of the task now, because after calling
   * g_task_return_pointer, the callback will be called in the
   * main thread through g_task_return_now, and then we can
   * unref the task here. */
  g_autoptr(GTask) task = user_data;
  g_autoptr(GGMLModel) model = NULL;
  g_auto(GStrv) loaded_keys = NULL;

  if ((model = ggml_model_load_from_istream_finish (result, &loaded_keys, &error)) == NULL)
    {
      g_task_return_error (task, error);
      return;
    }

  GGMLLanguageModelLoadFromIstreamData *data = g_task_get_task_data (task);

  data->model = g_steal_pointer (&model);

  const char *src_weights[] = {"model/wte", NULL};
  const char *dst_weights[] = {"model/lm_head", NULL};
  ggml_model_set_possible_tied_weights (data->model, (const char **) loaded_keys, src_weights, dst_weights);

  g_task_return_pointer (task,
                         ggml_language_model_new (data->hyperparameters,
                                                  data->token_dictionary,
                                                  data->model),
                         (GDestroyNotify) ggml_language_model_unref);
}

static void
ggml_language_model_load_from_istream_on_token_dictionary_read (GObject *src,
                                                                GAsyncResult *result,
                                                                gpointer user_data)
{
  GError *error = NULL;
  GTask *task = user_data;
  g_autoptr(GGMLTokenDictionary) token_dictionary = NULL;

  if ((token_dictionary = ggml_token_dictionary_load_from_istream_finish (result, &error)) == NULL)
    {
      g_task_return_error (task, error);
      return;
    }

  GGMLLanguageModelLoadFromIstreamData *data = g_task_get_task_data (task);

  data->token_dictionary = g_steal_pointer (&token_dictionary);

  /* Continue reading the stream, now for the model itself.
   *
   * After launching this, the model_forwad_func_user_data is transferred
   * to the subtask, so set to %NULL in the GGMLHyperparametersLoadFromIstreamData
   */
  ggml_model_load_from_istream_async (data->istream,
                                      data->model_desc,
                                      data->hyperparameters,
                                      g_steal_pointer (&data->forward_func),
                                      g_steal_pointer (&data->forward_func_user_data),
                                      g_steal_pointer (&data->forward_func_user_data_destroy),
                                      g_task_get_cancellable (task),
                                      ggml_language_model_load_from_istream_on_model_read,
                                      task);
}

static void
ggml_language_model_load_from_istream_on_hyperparameters_read (GObject *src,
                                                               GAsyncResult *result,
                                                               gpointer user_data)
{
  GError *error = NULL;
  GTask *task = user_data;
  g_autoptr(GGMLHyperparameters) hyperparameters = NULL;

  if ((hyperparameters = ggml_hyperparameters_load_from_istream_finish (result, &error)) == NULL)
    {
      g_task_return_error (task, error);
      return;
    }

  GGMLLanguageModelLoadFromIstreamData *data = g_task_get_task_data (task);

  /* We can already use the hyperparameters to create the model desc. */
  data->hyperparameters = g_steal_pointer (&hyperparameters);
  data->model_desc = (*data->create_model_desc) (data->hyperparameters,
                                                 data->create_model_desc_user_data);

  /* Continue reading the stream, now for the token dictionary */
  ggml_token_dictionary_load_from_istream_async (data->istream,
                                                 ggml_hyperparameters_get_int32 (data->hyperparameters, "n_vocab"),
                                                 g_task_get_cancellable (task),
                                                 ggml_language_model_load_from_istream_on_token_dictionary_read,
                                                 task);
}

static void
ggml_language_model_load_from_istream_on_magic_read (GObject *src,
                                                     GAsyncResult *result,
                                                     gpointer user_data)
{
  GError *error = NULL;
  GTask *task = user_data;

  if (!ggml_language_model_consume_istream_magic_finish (result, &error))
    {
      g_task_return_error (task, error);
      return;
    }

  /* Continue reading the istream */
  GGMLLanguageModelLoadFromIstreamData *data = g_task_get_task_data (task);

  ggml_hyperparameters_load_from_istream_async (data->istream,
                                                g_task_get_cancellable (task),
                                                ggml_language_model_load_from_istream_on_hyperparameters_read,
                                                task);
}

/**
 * ggml_language_model_load_from_istream_async:
 * @istream: (transfer none): A #GInputStream
 * @create_model_desc: (transfer none) (scope call): A #GGMLModelDescFromHyperparametersFunc to specify the model structure and weights
 * @create_model_desc_user_data: (closure create_model_desc): A closure for @create_model_desc
 * @create_model_desc_user_data_destroy: (destroy create_model_desc): A #GDestroyNotify for create_model_desc
 * @forward_func: (scope notified) (nullable): A #GGMLModelFowardFunc
 * @forward_func_user_data: (closure forward_func) (transfer full): The user data for @forward_func
 * @forward_func_user_data_destroy: (destroy forward_func): A #GDestroyNotify for forward_func
 * @cancellable: (transfer none) (nullable): A #GCancellable
 * @callback: A #GAsyncReadyCallback to be called when loading is complete.
 * @user_data: (closure callback): Some user data for @callback
 *
 * Asynchronously read a GGML language model from @istream , which includes the hyperparameters, token
 * dictionary and model weights. The @callback will be called with a #GGMLLanguageModel
 * or an error on completion.
 */
void
ggml_language_model_load_from_istream_async (GInputStream *istream,
                                             GGMLModelDescFromHyperparametersFunc create_model_desc,
                                             gpointer create_model_desc_user_data,
                                             GDestroyNotify create_model_desc_user_data_destroy,
                                             GGMLModelForwardFunc forward_func,
                                             gpointer forward_func_user_data,
                                             GDestroyNotify forward_func_user_data_destroy,
                                             GCancellable *cancellable,
                                             GAsyncReadyCallback callback,
                                             gpointer user_data)
{
  g_autoptr(GGMLLanguageModelLoadFromIstreamData) data = ggml_language_model_load_from_istream_data_new(istream,
                                                                                                        create_model_desc,
                                                                                                        create_model_desc_user_data,
                                                                                                        create_model_desc_user_data_destroy,
                                                                                                        forward_func,
                                                                                                        forward_func_user_data,
                                                                                                        forward_func_user_data_destroy);

  g_autoptr(GTask) task = g_task_new (NULL, cancellable, callback, user_data);
  g_task_set_task_data (task, g_steal_pointer (&data), (GDestroyNotify) ggml_language_model_load_from_istream_data_free);

  /* In this case, ggml_language_model_consume_istream_magic_async owns the parent task
   * so we steal the pointer from here. */
  ggml_language_model_consume_istream_magic_async (istream,
                                                   cancellable,
                                                   ggml_language_model_load_from_istream_on_magic_read,
                                                   g_steal_pointer (&task));
}

/**
 * ggml_language_model_load_from_istream_finish:
 * @result: A #GAsyncResult
 * @error: (nullable): A #GError
 *
 * Finish an async read of a #GGMLLanguageModel and return the model.
 *
 * Returns: (transfer full): A new #GGMLLanguageModel or %NULL with @error set
 *          on failure.
 */
GGMLLanguageModel *
ggml_language_model_load_from_istream_finish (GAsyncResult  *result,
                                              GError       **error)
{
  g_return_val_if_fail (g_task_is_valid (result, NULL), NULL);

  return g_task_propagate_pointer (G_TASK (result), error);
}

/**
 * ggml_language_model_load_defined_from_istream_finish:
 * @result: A #GAsyncResult
 * @error: (nullable): A #GError
 *
 * Finish an async read of a #GGMLLanguageModel and return the model.
 *
 * Returns: (transfer full): A new #GGMLLanguageModel or %NULL with @error set
 *          on failure.
 */
GGMLLanguageModel *
ggml_language_model_load_defined_from_istream_finish (GAsyncResult  *result,
                                                      GError       **error)
{
  g_return_val_if_fail (g_task_is_valid (result, NULL), NULL);

  return g_task_propagate_pointer (G_TASK (result), error);
}

/**
 * ggml_language_model_load_defined_from_istream_async:
 * @model: A #GGMLDefinedLanguageModel configuration to load
 * @istream: (transfer none): A #GInputStream
 * @cancellable: (transfer none) (nullable): A #GCancellable
 * @callback: A #GAsyncReadyCallback to be called when loading is complete.
 * @user_data: (closure callback): Some user data for @callback
 *
 * Asynchronously read a GGML language model from @istream , which includes the hyperparameters, token
 * dictionary and model weights. This is more language binding friendly because it does
 * not require calling back into the bindings to finalize model loading, but it is also more
 * inflexible because it is limited to a set of hardcoded models. The @callback will be called with a #GGMLLanguageModel
 * or an error on completion.
 */
void
ggml_language_model_load_defined_from_istream_async (GGMLDefinedLanguageModel   model,
                                                     GInputStream              *istream,
                                                     GCancellable              *cancellable,
                                                     GAsyncReadyCallback        callback,
                                                     gpointer                   user_data,
                                                     GError                   **error)
{
  ggml_language_model_load_from_istream_async (istream,
                                               ggml_language_model_definitions[model].model_desc_from_hyperparameters_func,
                                               NULL,
                                               NULL,
                                               ggml_language_model_definitions[model].forward_func,
                                               NULL,
                                               NULL,
                                               cancellable,
                                               callback,
                                               user_data);
}

/**
 * ggml_language_model_ref:
 * @language_model: A #GGMLLanguageModel
 *
 * Increments the ref count on @language_model
 *
 * Returns: (transfer full): A #GGMLLanguageModel
 */
GGMLLanguageModel *
ggml_language_model_ref (GGMLLanguageModel *language_model)
{
  ++language_model->ref_count;

  return language_model;
}

/**
 * ggml_language_model_unref:
 * @language_model: A #GGMLLanguageModel
 *
 * Decreases the ref count on @language_model . If the ref count goes to
 * zero, then the language model will be cleaned up. Note that the underlying
 * memory is not freed until the corresponding #GGMLContext in the #GGMLModel
 * is released and its memory pool is cleaned up.
 */
void
ggml_language_model_unref (GGMLLanguageModel *language_model)
{
  if (--language_model->ref_count == 0)
    {
      g_clear_pointer (&language_model->hyperparameters, ggml_hyperparameters_unref);
      g_clear_pointer (&language_model->token_dictionary, ggml_token_dictionary_unref);
      g_clear_pointer (&language_model->model, ggml_model_unref);
      g_clear_pointer (&language_model, g_free);
    }
}

G_DEFINE_BOXED_TYPE (GGMLLanguageModel, ggml_language_model, ggml_language_model_ref, ggml_language_model_unref);
