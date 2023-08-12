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

#include <ggml-gobject/ggml-cached-model.h>
#include <ggml-gobject/ggml-execution-memory.h>
#include <ggml-gobject/ggml-gpt.h>
#include <ggml-gobject/ggml-language-model.h>
#include <ggml-gobject/ggml-quantize.h>
#include <ggml-gobject/internal/ggml-async-queue-source.h>
#include <ggml-gobject/internal/ggml-stream-internal.h>

struct _GGMLLanguageModelCompletionCursor {
  GGMLLanguageModel *language_model;
  GGMLExecutionMemory *execution_memory;
  char *prompt;
  size_t max_completion_tokens;
  size_t memory_position;
  int32_t most_recent_token;
  gboolean is_executing;
  size_t ref_count;
};

GGMLLanguageModelCompletionCursor *
ggml_language_model_completion_cursor_ref (GGMLLanguageModelCompletionCursor *cursor)
{
  ++cursor->ref_count;
  return cursor;
}

void
ggml_language_model_completion_cursor_unref (GGMLLanguageModelCompletionCursor *cursor)
{
  if (--cursor->ref_count == 0)
    {
      g_clear_pointer (&cursor->language_model, ggml_language_model_unref);
      g_clear_pointer (&cursor->execution_memory, ggml_execution_memory_unref);
      g_clear_pointer (&cursor->prompt, g_free);
      g_clear_pointer (&cursor, g_free);
    }
}

G_DEFINE_BOXED_TYPE (GGMLLanguageModelCompletionCursor,
                     ggml_language_model_completion_cursor,
                     ggml_language_model_completion_cursor_ref,
                     ggml_language_model_completion_cursor_unref);


/**
 * ggml_language_model_desc_new:
 * @weights_desc_node: A #GGMLModelDescNode describing the the model weights
 * @memory_desc_node: (nullable): A #GGMLModelDescNode describing the memory weights
 *
 * Returns: (transfer full): A new #GGMLLanguageModelDescNode
 */
GGMLLanguageModelDesc *
ggml_language_model_desc_new (GGMLModelDescNode *weights_desc_node,
                              GGMLModelDescNode *memory_desc_node)
{
  GGMLLanguageModelDesc *language_model_desc = g_new0 (GGMLLanguageModelDesc, 1);
  language_model_desc->weights_desc = ggml_model_desc_node_ref (weights_desc_node);
  language_model_desc->memory_desc = memory_desc_node != NULL ? ggml_model_desc_node_ref (memory_desc_node) : NULL;

  return language_model_desc;
}

GGMLLanguageModelDesc *
ggml_language_model_desc_copy (GGMLLanguageModelDesc *language_model_desc)
{
  return ggml_language_model_desc_new (language_model_desc->weights_desc,
                                       language_model_desc->memory_desc);
}

void
ggml_language_model_desc_free (GGMLLanguageModelDesc *language_model_desc)
{
  g_clear_pointer (&language_model_desc->weights_desc, ggml_model_desc_node_unref);
  g_clear_pointer (&language_model_desc->memory_desc, ggml_model_desc_node_unref);
  g_clear_pointer (&language_model_desc, g_free);
}

G_DEFINE_BOXED_TYPE (GGMLLanguageModelDesc,
                     ggml_language_model_desc,
                     ggml_language_model_desc_copy,
                     ggml_language_model_desc_free)

struct _GGMLLanguageModel {
  GGMLHyperparameters *hyperparameters;
  GGMLTokenDictionary *token_dictionary;
  GGMLModel *model;
  GGMLModelDescNode *memory_desc_node;
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

  g_autoptr(GTask) task = G_TASK (result);

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
  GTask *task = g_task_new (NULL, cancellable, callback, user_data);
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
ggml_language_model_new (GGMLHyperparameters *hyperparameters,
                         GGMLTokenDictionary *dictionary,
                         GGMLModel           *model,
                         GGMLModelDescNode   *memory_desc_node)
{
  GGMLLanguageModel *language_model = g_new0 (GGMLLanguageModel, 1);
  language_model->hyperparameters = ggml_hyperparameters_ref (hyperparameters);
  language_model->token_dictionary = ggml_token_dictionary_ref (dictionary);
  language_model->model = ggml_model_ref (model);
  language_model->memory_desc_node = ggml_model_desc_node_ref (memory_desc_node);
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
                                              GGMLExecutionMemory  *execution_memory,
                                              int32_t              *input_tokens,
                                              size_t                n_input_tokens,
                                              GCancellable         *cancellable,
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
                                                            execution_memory,
                                                            cancellable,
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

typedef struct _GGMLLanguageModelCompleteState
{
  GGMLLanguageModelCompletionCursor *cursor;
  size_t iterations;
  size_t chunk_size;
  GAsyncQueue *queue;
  GCancellable *cancellable;
} GGMLLanguageModelCompleteState;

static GGMLLanguageModelCompleteState *
ggml_language_model_complete_state_new (GGMLLanguageModelCompletionCursor *cursor,
                                        size_t                             iterations,
                                        size_t                             chunk_size,
                                        GAsyncQueue                       *async_queue,
                                        GCancellable                      *cancellable)
{
  GGMLLanguageModelCompleteState *state = g_new0 (GGMLLanguageModelCompleteState, 1);
  state->cursor = ggml_language_model_completion_cursor_ref (cursor);
  state->iterations = iterations;
  state->chunk_size = chunk_size;
  state->queue = g_async_queue_ref (async_queue);
  state->cancellable = cancellable != NULL ? g_object_ref (cancellable) : NULL;

  return state;
}

static void
ggml_language_model_complete_state_free (GGMLLanguageModelCompleteState *state)
{
  g_clear_pointer (&state->cursor, ggml_language_model_completion_cursor_unref);
  g_clear_pointer (&state->cancellable, g_object_unref);
  g_clear_pointer (&state->queue, g_async_queue_unref);

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

  /* If this is the is_complete chunk, then we're no longer executing
   * and can remove the gate */
  if (is_complete == TRUE)
    {
      state->cursor->is_executing = FALSE;
    }
}

static gpointer
ggml_language_model_complete_cursor_thread_loop (gpointer data)
{
  GGMLLanguageModelCompleteState *state = data;
  g_autofree int32_t *out_prompt_tokens = NULL;
  size_t   out_n_prompt_tokens = 0;
  int32_t  n_completed_iterations = 0;
  g_autoptr(GError) error = NULL;
  g_autoptr(GHashTable) inference_parameters = NULL;

  /* Gate behind the is_executing variable. If we are already executing and
   * re-called this function, then we have to return an error */
  if (state->cursor->is_executing == TRUE)
    {
      g_autoptr(GError) error = g_error_new (G_IO_ERROR,
                                             G_IO_ERROR_FAILED,
                                             "Already executing on this cursor");
      ggml_language_model_complete_thread_push_tokens_or_error (state,
                                                                NULL,
                                                                FALSE,
                                                                FALSE,
                                                                g_steal_pointer (&error));
      return GINT_TO_POINTER (FALSE);
    }

  state->cursor->is_executing = TRUE;

  inference_parameters = g_hash_table_new_full (g_str_hash, g_str_equal, NULL , NULL);

  /* Allocate space for a chunk buffer to save processed tokens into */
  g_autoptr(GArray) chunk_tokens = g_array_sized_new (FALSE, TRUE, sizeof (int32_t), state->chunk_size);
  int32_t *chunk_tokens_data = &(g_array_index (chunk_tokens, int32_t, 0));

  /* We allocate enough space upfront and also set the size.
   * This is used as a ring buffer, so all zeros are fine */
  chunk_tokens->len = state->chunk_size;

  int32_t current_chunk_index = 0;

  if (state->cursor->execution_memory == NULL)
    {
      g_autoptr(GGMLExecutionMemory) recorder_execution_memory = ggml_execution_memory_recorder_new (state->cursor->language_model->memory_desc_node);

      /* Create an input with max_completion_tokens. We have to allocate
       * here because creating the variant will copy */
      g_autoptr(GArray) dummy_input_array = g_array_sized_new (FALSE,
                                                               TRUE,
                                                               sizeof (int32_t),
                                                               state->cursor->max_completion_tokens);
      g_array_set_size (dummy_input_array, state->cursor->max_completion_tokens);

      g_autoptr(GVariant) dummy_inputs = g_variant_ref_sink (g_variant_new_fixed_array (G_VARIANT_TYPE_INT32,
                                                                                        dummy_input_array->data,
                                                                                        state->cursor->max_completion_tokens,
                                                                                        sizeof (int32_t)));

      /* In this case, n_past is always zero */
      g_hash_table_insert (inference_parameters,
                           (gpointer) n_past_key,
                           GINT_TO_POINTER (state->cursor->memory_position));

      /* We must first do a worst-case pass through the model to
       * determine what the real exection memory usage is */
      g_autoptr(GGMLTensor) output_tensor = NULL;
      g_autoptr(GGMLComputeGraph) compute_graph = ggml_model_build_graph (
        state->cursor->language_model->model,
        state->cursor->language_model->hyperparameters,
        dummy_inputs,
        inference_parameters,
        recorder_execution_memory,
        &output_tensor,
        &error
      );

      if (compute_graph == NULL)
        {
          ggml_language_model_complete_thread_push_tokens_or_error (state,
                                                                    NULL,
                                                                    FALSE,
                                                                    FALSE,
                                                                    g_steal_pointer (&error));
          return GINT_TO_POINTER (FALSE);
        }

      size_t execution_memory_size = ggml_compute_graph_get_computation_size (compute_graph,
                                                                              output_tensor);

      g_autoptr(GHashTable) flattened_memory_desc = ggml_model_desc_node_flatten (state->cursor->language_model->memory_desc_node);
      g_autoptr(GHashTable) memory_weight_set = ggml_new_weight_set_from_flattened_desc (NULL, flattened_memory_desc);

      state->cursor->execution_memory = ggml_execution_memory_new (
        execution_memory_size,
        memory_weight_set
      );
    }

  for (; n_completed_iterations < state->iterations; ++n_completed_iterations)
    {
      int32_t *forward_input_tokens_ptr = NULL;
      size_t n_forward_input_tokens = 0;

      g_hash_table_insert (inference_parameters,
                           (gpointer) n_past_key,
                           GINT_TO_POINTER (state->cursor->memory_position));

      if (state->cursor->memory_position == 0)
        {
          /* First iteration, we have to initially tokenize and seed the memory */
          if (!ggml_gpt_tokenize (state->cursor->language_model->token_dictionary,
                                  state->cursor->prompt,
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
          g_autofree char *init_chunk = g_strdup (state->cursor->prompt);
          /* We completed a chunk, send it to the caller. */
          ggml_language_model_complete_thread_push_tokens_or_error (state,
                                                                    g_steal_pointer (&init_chunk),
                                                                    FALSE,
                                                                    FALSE,
                                                                    NULL);

          /* Set the forward_input_tokens_ptr to the tokenized tokens */
          forward_input_tokens_ptr = out_prompt_tokens;
          n_forward_input_tokens = out_n_prompt_tokens;
        }
      else
        {
          /* Set the forward_input_tokens_ptr to the tokenized tokens */
          forward_input_tokens_ptr = &state->cursor->most_recent_token;
          n_forward_input_tokens = 1;
        }

      if (!ggml_language_model_forward_single_iteration (state->cursor->language_model->model,
                                                         state->cursor->language_model->hyperparameters,
                                                         inference_parameters,
                                                         state->cursor->execution_memory,
                                                         forward_input_tokens_ptr,
                                                         n_forward_input_tokens,
                                                         state->cancellable,
                                                         &state->cursor->most_recent_token,
                                                         &error))
        {
          ggml_language_model_complete_thread_push_tokens_or_error (state,
                                                                    NULL,
                                                                    FALSE,
                                                                    FALSE,
                                                                    g_steal_pointer (&error));
          return GINT_TO_POINTER (FALSE);
        }

      chunk_tokens_data[current_chunk_index++] = state->cursor->most_recent_token;

      if (current_chunk_index == state->chunk_size)
        {
          g_autofree char *chunk = ggml_token_dictionary_decode (state->cursor->language_model->token_dictionary,
                                                                 chunk_tokens_data,
                                                                 chunk_tokens->len);
          /* We completed a chunk, send it to the caller. */
          ggml_language_model_complete_thread_push_tokens_or_error (state,
                                                                    g_steal_pointer (&chunk),
                                                                    FALSE,
                                                                    FALSE,
                                                                    NULL);
          current_chunk_index = 0;
        }

      /* Increment by num_forward_input_tokens - this is the number of tokens
       * we had to process and add to the memory */
      state->cursor->memory_position += n_forward_input_tokens;
    }

  g_array_set_size (chunk_tokens, current_chunk_index);
  g_autofree char *chunk = ggml_token_dictionary_decode (state->cursor->language_model->token_dictionary,
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
  GGMLLanguageModelCompletionCursorStreamFunc stream_func;
  gpointer stream_func_data;
  GDestroyNotify stream_func_data_destroy;
  GAsyncReadyCallback callback;
  gpointer user_data;
  size_t ref_count;
} GGMLLanguageModelCompleteMonitorState;

static GGMLLanguageModelCompleteMonitorState *
ggml_language_model_complete_monitor_state_new (GGMLLanguageModelCompletionCursorStreamFunc stream_func,
                                                gpointer                                    stream_func_data,
                                                GDestroyNotify                              stream_func_data_destroy,
                                                GAsyncReadyCallback                         callback,
                                                gpointer                                    user_data)
{
  GGMLLanguageModelCompleteMonitorState *state = g_new0 (GGMLLanguageModelCompleteMonitorState, 1);
  state->stream_func = stream_func;
  state->stream_func_data = stream_func_data;
  state->stream_func_data_destroy = stream_func_data_destroy;
  state->callback = callback;
  state->user_data = user_data;
  state->ref_count = 1;

  return state;
}

static void
ggml_language_model_complete_monitor_state_unref (GGMLLanguageModelCompleteMonitorState *state)
{
  if (--state->ref_count == 0)
    {
      /* Assuming here that we will have called the callback
       * at the end */
      if (state->stream_func_data_destroy != NULL)
        {
          g_clear_pointer (&state->stream_func_data, state->stream_func_data_destroy);
        }

      g_clear_pointer (&state, g_free);
    }
}

static GGMLLanguageModelCompleteMonitorState *
ggml_language_model_complete_monitor_state_ref (GGMLLanguageModelCompleteMonitorState *state)
{
  ++state->ref_count;
  return state;
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLLanguageModelCompleteMonitorState, ggml_language_model_complete_monitor_state_unref)

static void
ggml_language_model_monitor_process_completion_ready (GObject      *source_object,
                                                      GAsyncResult *result,
                                                      gpointer      user_data)
{
  g_autoptr(GGMLLanguageModelCompleteMonitorState) state = user_data;
  g_autoptr(GTask) task = G_TASK (result);

  /* We call the real callback. That will also want to unref the task
   * so we have to ref it before calling the callback. */
  (*state->callback) (source_object, G_ASYNC_RESULT (g_object_ref (task)), state->user_data);
}

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
  if (completion->error != NULL)
    {
      GTask *task = g_task_new (NULL,
                                NULL,
                                ggml_language_model_monitor_process_completion_ready,
                                ggml_language_model_complete_monitor_state_ref (state));

      GError *error = NULL;
      g_propagate_error (&error, g_steal_pointer (&completion->error));
      g_task_return_error (task, error);
      return TRUE;
    }

  if (completion->result != NULL)
    {
      (*state->stream_func) (completion->result->chunk,
                             completion->result->is_complete_eos,
                             state->stream_func_data);

      if (completion->result->is_complete)
        {
          /* We got a sentinel value. This means that we can now complete
           * the task. We have to use an extra trampoline with a ref on the
           * GGMLLanguageModelCompleteMonitorState here because
           * g_task_return_error / g_task_return_pointer could
           * end up on the main loop for another iteration. */
          GTask *task = g_task_new (NULL,
                                    NULL,
                                    ggml_language_model_monitor_process_completion_ready,
                                    ggml_language_model_complete_monitor_state_ref (state));

          g_task_return_boolean (task, TRUE);

          /* We're all done, return TRUE because we got a sentinel value */
          return TRUE;
        }

      return FALSE;
    }

  g_assert_not_reached ();
  return TRUE;
}

static gboolean
ggml_language_model_monitor_callback (gpointer message, gpointer user_data)
{
  GGMLLanguageModelCompleteMonitorState *state = user_data;
  g_autoptr(GGMLLanguageModelChunkCompletion) completion = message;

  /* If we return TRUE*/
  if (ggml_language_model_monitor_process_completion (state, completion))
    {
      return G_SOURCE_REMOVE;
    }

  return G_SOURCE_CONTINUE;
}

/**
 * ggml_language_model_create_completion:
 * @language_model: A #GGMLLanguageModel
 * @prompt: (transfer none): A text prompt to seed the language model
 * @max_completion_tokens: Maximum number of tokens in this query. Generating any more
 *                         requires creating a new cursor.
 *
 * Returns: (transfer full): A new #GGMLLanguageModelCompletionCursor representing the
 *          initial state of execution of the language model evaluated on some prompt.
 */
GGMLLanguageModelCompletionCursor *
ggml_language_model_create_completion (GGMLLanguageModel *language_model,
                                       const char        *prompt,
                                       size_t             max_completion_tokens)
{
  GGMLLanguageModelCompletionCursor *cursor = g_new0 (GGMLLanguageModelCompletionCursor, 1);
  cursor->language_model = ggml_language_model_ref (language_model);
  cursor->execution_memory = NULL;
  cursor->prompt = g_strdup (prompt);
  cursor->max_completion_tokens = max_completion_tokens;
  cursor->memory_position = 0;
  cursor->ref_count = 1;

  return cursor;
}

/**
 * ggml_language_model_completion_cursor_exec_stream_async:
 * @cursor: (transfer none): A #GGMLLanguageModelCompletionCursor
 * @num_iterations: Number of additional tokens to generate
 * @stream_chunk_size: Chunk size of tokens that get sent to @callback on generation
 * @cancellable: (transfer none) (nullable): A #GCancellable
 * @stream_func: A #GGMLLanguageModelCompletionCursorStreamFunc to stream the results to
 * @stream_func_data: (closure stream_func): User data for the @stream_func
 * @stream_func_data_destroy: (destroy stream_func) (nullable): A #GDestroyNotify for @stream_func_data
 * @callback: A #GAsyncReadyCallback called once the operation is complete
 * @user_data: (closure callback): Some user data for @callback
 *
 * Asynchronously execute from from @cursor by generating @num_iterations tokens.
 * This function can be used for streaming the result of generation, if @chunk_size < @num_iterations
 * as @stream_func will be invoked multiple times. @chunk_size is something that will need to be tuned according
 * to the needs of the application. A smaller @chunk_size means lower latency but higher overhead and
 * therefore slower overall generation.
 *
 * The @stream_func should not free its @stream_func_data on
 * invocation - the generation process will do that once generation is complete. You can complete
 * a call to this function with ggml_language_model_completion_cursor_complete_exec_stream from
 * within @callback.
 */
void
ggml_language_model_completion_cursor_exec_stream_async (GGMLLanguageModelCompletionCursor           *cursor,
                                                         size_t                                       num_iterations,
                                                         size_t                                       stream_chunk_size,
                                                         GCancellable                                *cancellable,
                                                         GGMLLanguageModelCompletionCursorStreamFunc  stream_func,
                                                         gpointer                                     stream_func_data,
                                                         GDestroyNotify                               stream_func_data_destroy,
                                                         GAsyncReadyCallback                          callback,
                                                         gpointer                                     user_data)
{
  /* We don't pass the cancellable to the task, but instead to
   * the GGMLLanguageModelCompleteState . The reason is that the task
   * is only there as a crutch to be used as a GAsyncResult
   * and */
  g_autoptr(GError) error = NULL;

  g_autoptr(GAsyncQueue) async_queue = g_async_queue_new_full ((GDestroyNotify) ggml_language_model_chunk_completion_free);
  g_autoptr(GGMLLanguageModelCompleteMonitorState) monitor_state = ggml_language_model_complete_monitor_state_new (stream_func,
                                                                                                                   stream_func_data,
                                                                                                                   stream_func_data_destroy,
                                                                                                                   callback,
                                                                                                                   user_data);

  GSource *monitor_source = ggml_async_queue_source_new (async_queue,
                                                         ggml_language_model_monitor_callback,
                                                         g_steal_pointer (&monitor_state),
                                                         (GDestroyNotify) ggml_language_model_complete_monitor_state_unref,
                                                         cancellable);
  g_source_attach (g_steal_pointer (&monitor_source), NULL);

  GGMLLanguageModelCompleteState *state = ggml_language_model_complete_state_new (cursor,
                                                                                  num_iterations,
                                                                                  stream_chunk_size,
                                                                                  async_queue,
                                                                                  cancellable);

  g_autoptr(GThread) thread = g_thread_new ("complete-thread", ggml_language_model_complete_cursor_thread_loop, state);
}

/**
 * ggml_language_model_completion_cursor_exec_stream_finish:
 * @cursor: (transfer none): A #GGMLLanguageModelCompletionCursor
 * @result: A #GAsyncResult that came from the callback
 * @error: A #GError out-parameter
 *
 * Complete the call to %ggml_language_model_completion_cursor_exec_stream_async .Note that this
 * doesn't contain the result of the execution - that was streamed to the stream_func already.
 *
 * Returns: %TRUE the execution was successful, %FALSE with @error set if an error was encountered.
 */
gboolean
ggml_language_model_completion_cursor_exec_stream_finish (GGMLLanguageModelCompletionCursor  *cursor,
                                                          GAsyncResult                       *result,
                                                          GError                            **error)
{
  GTask *task = G_TASK (result);
  gboolean exec_result = g_task_propagate_boolean (task, error);

  return exec_result;
}

typedef struct _GGMLLanguageModelCompletionCursorStreamCollector
{
  GPtrArray *collected_chunks;
  gboolean is_complete_eos;
  GAsyncReadyCallback original_callback;
  gpointer original_data;
  size_t ref_count;
} GGMLLanguageModelCompletionCursorStreamCollector;

GGMLLanguageModelCompletionCursorStreamCollector *
ggml_language_model_completion_cursor_stream_collector_new (GAsyncReadyCallback original_callback,
                                                            gpointer            original_data)
{
  GGMLLanguageModelCompletionCursorStreamCollector *collector = g_new0 (GGMLLanguageModelCompletionCursorStreamCollector, 1);
  collector->original_callback = original_callback;
  collector->original_data = original_data;
  collector->collected_chunks = g_ptr_array_new_full (1, g_free);
  collector->ref_count = 1;

  return collector;
}

GGMLLanguageModelCompletionCursorStreamCollector *
ggml_language_model_completion_cursor_stream_collector_ref (GGMLLanguageModelCompletionCursorStreamCollector *collector)
{
  ++collector->ref_count;
  return collector;
}

void
ggml_language_model_completion_cursor_stream_collector_unref (GGMLLanguageModelCompletionCursorStreamCollector *collector)
{
  if (--collector->ref_count == 0)
    {
      g_clear_pointer (&collector->collected_chunks, g_ptr_array_unref);
      g_clear_pointer (&collector, g_free);
    }
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLLanguageModelCompletionCursorStreamCollector, ggml_language_model_completion_cursor_stream_collector_unref)

static void
collect_stream_func (const char *decoded,
                     gboolean    is_complete_eos,
                     gpointer    user_data)
{
  GGMLLanguageModelCompletionCursorStreamCollector *collector = user_data;

  collector->is_complete_eos |= is_complete_eos;
  g_ptr_array_add (collector->collected_chunks, g_strdup (decoded));
}

typedef struct {
  char *completion;
  gboolean is_complete_eos;
} GGMLLanguageModelCompletionCursorExecResult;

static GGMLLanguageModelCompletionCursorExecResult *
ggml_language_model_completion_cursor_exec_result_new (const char *completion,
                                                       gboolean    is_complete_eos)
{
  GGMLLanguageModelCompletionCursorExecResult *result = g_new0 (GGMLLanguageModelCompletionCursorExecResult, 1);
  result->completion = g_strdup (completion);
  result->is_complete_eos = is_complete_eos;

  return result;
}

static void
ggml_language_model_completion_cursor_exec_result_free (GGMLLanguageModelCompletionCursorExecResult *result)
{
  g_clear_pointer (&result->completion, g_free);
  g_clear_pointer (&result, g_free);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLLanguageModelCompletionCursorExecResult, ggml_language_model_completion_cursor_exec_result_free);

static void
completion_cursor_exec_async_wrapper_callback (GObject      *source_object,
                                               GAsyncResult *result,
                                               gpointer      user_data)
{
  GError *error = NULL;
  g_autoptr(GGMLLanguageModelCompletionCursorStreamCollector) collector = user_data;
  GTask *task = g_task_new (NULL,
                            NULL,
                            collector->original_callback,
                            collector->original_data);

  if (!ggml_language_model_completion_cursor_exec_stream_finish (NULL, result, &error))
    {
      g_task_return_error (task, g_steal_pointer (&error));
      return;
    }

  g_ptr_array_add (collector->collected_chunks, NULL);
  g_autofree gchar *completion = g_strjoinv ("", (char **) collector->collected_chunks->pdata);
  GGMLLanguageModelCompletionCursorExecResult *exec_result = ggml_language_model_completion_cursor_exec_result_new (completion,
                                                                                                               collector->is_complete_eos);
  g_task_return_pointer (task, g_steal_pointer (&exec_result), (GDestroyNotify) ggml_language_model_completion_cursor_exec_result_free);
}

const static size_t DEFAULT_STREAM_CHUNK_SIZE = 128;

/**
 * ggml_language_model_completion_cursor_exec_async:
 * @cursor: (transfer none): A #GGMLLanguageModelCompletionCursor
 * @num_iterations: Number of additional tokens to generate
 * @cancellable: (transfer none) (nullable): A #GCancellable
 * @callback: A #GAsyncReadyCallback called once the operation is complete
 * @user_data: (closure callback): Some user data for @callback
 *
 * Asynchronously execute from from @cursor by generating @num_iterations tokens.
 * This is a simpler API than %ggml_language_model_completion_cursor_exec_stream_async which
 * only returns the completion up to %num_iterations once it is actually done (without any streaming),
 * which means that for an application it may be higher-latency.
 */
void
ggml_language_model_completion_cursor_exec_async (GGMLLanguageModelCompletionCursor           *cursor,
                                                  size_t                                       num_iterations,
                                                  GCancellable                                *cancellable,
                                                  GAsyncReadyCallback                          callback,
                                                  gpointer                                     user_data)
{
  g_autoptr(GGMLLanguageModelCompletionCursorStreamCollector) collector = ggml_language_model_completion_cursor_stream_collector_new (callback, user_data);

  /* We transfer the collector to both the stream_func callback and the task callback */
  ggml_language_model_completion_cursor_exec_stream_async (cursor,
                                                           num_iterations,
                                                           DEFAULT_STREAM_CHUNK_SIZE,
                                                           cancellable,
                                                           collect_stream_func,
                                                           ggml_language_model_completion_cursor_stream_collector_ref (collector),
                                                           (GDestroyNotify) ggml_language_model_completion_cursor_stream_collector_unref,
                                                           completion_cursor_exec_async_wrapper_callback,
                                                           ggml_language_model_completion_cursor_stream_collector_ref (collector));
}

/**
 * ggml_language_model_completion_cursor_exec_finish:
 * @cursor: (transfer none): A #GGMLLanguageModelCompletionCursor
 * @result: A #GAsyncResult that came from the callback
 * @out_is_complete_eos: (out): An out-param indicating that whether we reached an
 *                       end-of-sentence token.
 * @error: A #GError out-parameter
 *
 * Complete the call to %ggml_language_model_completion_cursor_exec_async
 *
 * Returns: (transfer full): The completed string, including the prompt or %NULL with @error set if
 *          execution was not successful.
 */
char *
ggml_language_model_completion_cursor_exec_finish (GGMLLanguageModelCompletionCursor  *cursor,
                                                   GAsyncResult                       *result,
                                                   gboolean                           *out_is_complete_eos,
                                                   GError                            **error)
{
  g_autoptr(GTask) task = G_TASK (result);
  g_autoptr(GGMLLanguageModelCompletionCursorExecResult) exec_result = NULL;

  exec_result = g_task_propagate_pointer (task, error);

  if (exec_result == NULL)
    {
      return NULL;
    }

  if (out_is_complete_eos != NULL)
    {
      *out_is_complete_eos = exec_result->is_complete_eos;
    }

  return g_steal_pointer (&exec_result->completion);
}

/**
 * ggml_language_model_completion_cursor_exec:
 * @cursor: (transfer none): A #GGMLLanguageModelCompletionCursor
 * @num_iterations: Number of additional tokens to generate
 * @cancellable: (transfer none) (nullable): A #GCancellable
 * @out_is_complete_eos: (out): An out-param indicating that whether we reached an
 *                       end-of-sentence token.
 * @error: A #GError out-parameter
 *
 * Synchronously execute from from @cursor by generating @num_iterations tokens. The operation
 * will block until all num_iterations have been run - this may be quite costly and not
 * suitable for interactive applications, as the process of running the model may take
 * many seconds.
 *
 * Returns: (transfer full): A string with the completion on success, or %NULL with
 *          @error set on failure.
 */
char *
ggml_language_model_completion_cursor_exec (GGMLLanguageModelCompletionCursor  *cursor,
                                            int32_t                             num_iterations,
                                            GCancellable                       *cancellable,
                                            gboolean                           *out_is_complete_eos,
                                            GError                            **error)
{
  g_autoptr(GAsyncQueue) async_queue = g_async_queue_new_full ((GDestroyNotify) ggml_language_model_chunk_completion_free);
  g_autoptr(GGMLLanguageModelCompleteState) state = ggml_language_model_complete_state_new (cursor,
                                                                                            num_iterations,
                                                                                            DEFAULT_STREAM_CHUNK_SIZE,
                                                                                            async_queue,
                                                                                            cancellable);

  /* Execute synchronously on the main thread */
  ggml_language_model_complete_cursor_thread_loop (state);

  g_autoptr(GPtrArray) completions_ptr_array = g_ptr_array_new_full (g_async_queue_length (async_queue) + 1, g_free);

  gboolean is_complete_eos = FALSE;

  /* Now we drain the async_queue and form a strv with completion chunks */
  while (g_async_queue_length (async_queue) > 0)
    {
      g_autoptr(GGMLLanguageModelChunkCompletion) completion = g_async_queue_try_pop (async_queue);
      g_assert (completion != NULL);

      if (completion->error != NULL)
        {
          g_propagate_error (error, g_steal_pointer (&completion->error));
          return NULL;
        }

      if (completion->result != NULL)
        {
          g_ptr_array_add (completions_ptr_array, g_steal_pointer (&completion->result->chunk));
          is_complete_eos |= completion->result->is_complete_eos;

          if (completion->result->is_complete)
            {
              /* Sentinel value */
              g_ptr_array_add (completions_ptr_array, NULL);
              break;
            }

          continue;
        }

      break;
    }

  *out_is_complete_eos = is_complete_eos;
  return g_strjoinv ("", (char **) completions_ptr_array->pdata);
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
 * @model_config: (nullable): A #GGMLModelConfig
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
                                       GGMLModelConfig *model_config,
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

  g_autoptr (GGMLLanguageModelDesc) language_model_desc = (*create_model_desc) (hyperparameters,
                                                                                create_model_desc_user_data);

  GGMLDataType quantized_type;
  const char **quantize_regexes = NULL;
  const char **skip_quantize_regexes = NULL;
  gboolean should_quantize = ggml_model_config_get_quantization_config (model_config,
                                                                        &quantized_type,
                                                                        &quantize_regexes,
                                                                        &skip_quantize_regexes);

  g_autoptr(GGMLModelDescNode) postprocessed_model_desc_node = (
    should_quantize ? ggml_configure_quantized_model_desc_by_regexes (language_model_desc->weights_desc,
                                                                      quantized_type,
                                                                      quantize_regexes,
                                                                      skip_quantize_regexes,
                                                                      error) :
                      ggml_model_desc_node_ref (language_model_desc->weights_desc)
  );

  if (postprocessed_model_desc_node == NULL)
    {
      return NULL;
    }

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
                                                             postprocessed_model_desc_node,
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
                                  model,
                                  language_model_desc->memory_desc);
}

static struct GGMLLanguageModelDefinitions {
  GGMLModelDescFromHyperparametersFunc model_desc_from_hyperparameters_func;
  GGMLModelForwardFunc forward_func;
} ggml_language_model_definitions[] = {
  /* GGML_DEFINED_MODEL_GPT2P117M */
  {
    .model_desc_from_hyperparameters_func = (GGMLModelDescFromHyperparametersFunc) ggml_create_gpt2_model_desc_from_hyperparameters,
    .forward_func = ggml_gpt_model_forward_pass
  },
  /* GGML_DEFINED_MODEL_GPT2P345M */
  {
    .model_desc_from_hyperparameters_func = (GGMLModelDescFromHyperparametersFunc) ggml_create_gpt2_model_desc_from_hyperparameters,
    .forward_func = ggml_gpt_model_forward_pass
  },
  /* GGML_DEFINED_MODEL_GPT2P774M */
  {
    .model_desc_from_hyperparameters_func = (GGMLModelDescFromHyperparametersFunc) ggml_create_gpt2_model_desc_from_hyperparameters,
    .forward_func = ggml_gpt_model_forward_pass
  },
  /* GGML_DEFINED_MODEL_GPT2P1558M */
  {
    .model_desc_from_hyperparameters_func = (GGMLModelDescFromHyperparametersFunc) ggml_create_gpt2_model_desc_from_hyperparameters,
    .forward_func = ggml_gpt_model_forward_pass
  }
};

/**
 * ggml_language_model_load_defined_from_istream:
 * @model: A #GGMLDefinedLanguageModel configuration to load
 * @istream: (transfer none): A #GInputStream
 * @model_config: (transfer none) (nullable): A #GGMLModelConfig
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
                                               GGMLModelConfig            *model_config,
                                               GCancellable               *cancellable,
                                               GError                    **error)
{
  return ggml_language_model_load_from_istream (istream,
                                                model_config,
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
  GGMLModelConfig *config;
  GGMLModelDescFromHyperparametersFunc create_model_desc;
  gpointer create_model_desc_user_data;
  GDestroyNotify create_model_desc_user_data_destroy;
  GGMLModelForwardFunc forward_func;
  gpointer forward_func_user_data;
  GDestroyNotify forward_func_user_data_destroy;

  /* Things that get loaded as we go */
  GGMLModelDescNode *model_desc;
  GGMLModelDescNode *memory_desc_node;
  GGMLHyperparameters *hyperparameters;
  GGMLTokenDictionary *token_dictionary;
  GGMLModel *model;
} GGMLLanguageModelLoadFromIstreamData;

static GGMLLanguageModelLoadFromIstreamData *
ggml_language_model_load_from_istream_data_new (GInputStream *istream,
                                                GGMLModelConfig *config,
                                                GGMLModelDescFromHyperparametersFunc create_model_desc,
                                                gpointer create_model_desc_user_data,
                                                GDestroyNotify create_model_desc_user_data_destroy,
                                                GGMLModelForwardFunc forward_func,
                                                gpointer forward_func_user_data,
                                                GDestroyNotify forward_func_user_data_destroy)
{
  GGMLLanguageModelLoadFromIstreamData *data = g_new0 (GGMLLanguageModelLoadFromIstreamData, 1);

  data->istream = g_object_ref (istream);
  data->config = config != NULL ? ggml_model_config_ref (config) : NULL;
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
  g_clear_pointer (&data->config, ggml_model_config_unref);
  g_clear_pointer (&data->create_model_desc_user_data, data->create_model_desc_user_data_destroy);
  g_clear_pointer (&data->forward_func_user_data, data->forward_func_user_data_destroy);

  g_clear_pointer (&data->memory_desc_node, ggml_model_desc_node_unref);
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
                                                  data->model,
                                                  data->memory_desc_node),
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
   * After launching this, the model_forward_func_user_data is transferred
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
  g_autoptr(GGMLLanguageModelDesc) language_model_desc = (*data->create_model_desc) (data->hyperparameters,
                                                                                     data->create_model_desc_user_data);
  data->memory_desc_node = g_steal_pointer (&language_model_desc->memory_desc);

  GGMLDataType quantized_type;
  const char **quantize_regexes = NULL;
  const char **skip_quantize_regexes = NULL;
  gboolean should_quantize = ggml_model_config_get_quantization_config (data->config,
                                                                        &quantized_type,
                                                                        &quantize_regexes,
                                                                        &skip_quantize_regexes);

  g_autoptr(GGMLModelDescNode) postprocessed_model_desc_node = (
    should_quantize ? ggml_configure_quantized_model_desc_by_regexes (language_model_desc->weights_desc,
                                                                      quantized_type,
                                                                      quantize_regexes,
                                                                      skip_quantize_regexes,
                                                                      &error) :
                      ggml_model_desc_node_ref (language_model_desc->weights_desc)
  );

  if (postprocessed_model_desc_node == NULL)
    {
      g_task_return_error (task, error);
      return;
    }

  data->model_desc = g_steal_pointer (&postprocessed_model_desc_node);

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
 * @model_config: (nullable): A #GGMLModelConfig
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
                                             GGMLModelConfig *model_config,
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
                                                                                                        model_config,
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
 * @model_config: (nullable): A #GGMLModelConfig
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
                                                     GGMLModelConfig           *model_config,
                                                     GCancellable              *cancellable,
                                                     GAsyncReadyCallback        callback,
                                                     gpointer                   user_data)
{
  ggml_language_model_load_from_istream_async (istream,
                                               model_config,
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

static const char *ggml_language_model_urls[] = {
  "https://huggingface.co/ggerganov/ggml/resolve/main/ggml-model-gpt-2-117M.bin",
  "https://huggingface.co/ggerganov/ggml/resolve/main/ggml-model-gpt-2-345M.bin",
  "https://huggingface.co/ggerganov/ggml/resolve/main/ggml-model-gpt-2-774M.bin",
  "https://huggingface.co/ggerganov/ggml/resolve/main/ggml-model-gpt-2-1558M.bin"
};

#define GGML_GOBJECT_MODELS_VERSION "0"

/**
 * ggml_language_model_stream_from_cache:
 * @defined_model: A #GGMLDefinedLanguageModel
 * @error: A #GError
 *
 * Creates a new #GGMLCachedModelIstream which will either download the model upon the first
 * read, or return a cached version from the disk.
 *
 * Returns: (transfer full): A #GGMLCachedModelIstream on success, %NULL with @error set on failure.
 */
GGMLCachedModelIstream *
ggml_language_model_stream_from_cache (GGMLDefinedLanguageModel   defined_model,
                                       GError                   **error)
{
  const char *remote_url = ggml_language_model_urls[defined_model];
  g_autofree char *path = NULL;

  if (!g_uri_split (remote_url, G_URI_FLAGS_NONE, NULL, NULL, NULL, NULL, &path, NULL, NULL, error))
    return NULL;

  g_autofree char *basename = g_path_get_basename (path);
  g_autofree char *local_path = g_build_filename (g_get_user_data_dir (),
                                                  "ggml-gobject",
                                                  GGML_GOBJECT_MODELS_VERSION,
                                                  "models",
                                                  basename,
                                                  NULL);

  return ggml_cached_model_istream_new (remote_url, local_path);
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
      g_clear_pointer (&language_model->memory_desc_node, ggml_model_desc_node_unref);
      g_clear_pointer (&language_model, g_free);
    }
}

G_DEFINE_BOXED_TYPE (GGMLLanguageModel, ggml_language_model, ggml_language_model_ref, ggml_language_model_unref);
