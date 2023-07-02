/*
 * ggml-gobject/ggml-gpt.c
 *
 * Library code for ggml-gpt
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
#include <math.h>

#define GPT_SPLIT_REGEX "('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\\s[:alpha:][:digit:]]+|\\s+(?!\\S)|\\s+)"

GPtrArray *
ggml_iterate_words_in_regex (GRegex *regex, const char *string)
{
  g_autoptr(GMatchInfo) match_info = NULL;
  g_autoptr(GPtrArray) words_ptr_array = g_ptr_array_new_full (0, g_free);
  g_regex_match (regex, string, 0, &match_info);
  while (g_match_info_matches (match_info))
    {
      gchar *word = g_match_info_fetch (match_info, 0);
      if (word != NULL)
        {
          g_ptr_array_add (words_ptr_array, word);
        }
      g_match_info_next (match_info, NULL);
    }

  return g_steal_pointer (&words_ptr_array);
}

/**
 * ggml_gpt_tokenize:
 * @token_dictionary: A #GGMLTokenDictionary of tokens
 * @string: A string to tokenize
 * @out_tokens: (out) (array length=out_size): Output tokens from the string
 * @error: A #GError
 *
 * Returns: %TRUE with @out_tokens and @out_size set on success, %FALSE
 *          with @error set otherwise.
 */
gboolean
ggml_gpt_tokenize (GGMLTokenDictionary *token_dictionary,
                   const char *string,
                   int32_t **out_tokens,
                   size_t  *out_size,
                   GError **error)
{
  /* Split first into words */
  g_autoptr(GArray) tokens_array = NULL;
  g_autoptr(GRegex) regex = NULL;
  g_autoptr(GPtrArray) words_ptr_array = NULL;

  regex = g_regex_new (GPT_SPLIT_REGEX,
                       0,
                       0,
                       error);

  if (regex == NULL)
    {
      return FALSE;
    }

  words_ptr_array = ggml_iterate_words_in_regex (regex, string);

  /* Now we have to find corresponding tokens in the dictionary */
  tokens_array = g_array_sized_new (FALSE,
                                    TRUE,
                                    sizeof (int32_t),
                                    words_ptr_array->len);

  for (size_t i = 0; i < words_ptr_array->len; ++i)
    {
      const char *word = words_ptr_array->pdata[i];
      size_t word_len = strlen(word);

      for (size_t word_start = 0; word_start < word_len;)
        {
          for (size_t word_end = word_len - 1;
               word_end >= word_start;
               --word_end)
            {
              /* Can't use autofree here because we're in a loop */
              char *candidate = g_strndup (&word[word_start],
                                           word_end - word_start + 1);
              int32_t token = 0;

              if (ggml_token_dictionary_lookup_extended (token_dictionary,
                                                         candidate,
                                                         &token))
                {
                  g_array_append_vals (tokens_array, &token, 1);
                  word_start = word_end + 1;
                  g_free (candidate);
                  break;
                }
              else if (word_end == word_start)
                {
                  ++word_start;
                  g_free (candidate);
                  break;
                }

              g_free (candidate);
            }
        }
    }

  *out_tokens = g_array_steal (tokens_array, out_size);
  return TRUE;
}

GGMLTensor *
ggml_nn_linear_layer (GGMLContext *context,
                      GGMLTensor *input,
                      GGMLTensor *weight,
                      GGMLTensor *bias)
{
  g_autoptr(GGMLTensor) weight_mul_output = ggml_op_mul_mat (context, weight, input);

  if (bias == NULL)
    {
      return g_steal_pointer (&weight_mul_output);
    }

  g_autoptr(GGMLTensor) repeat_bias = ggml_op_repeat (context, bias, weight_mul_output);
  g_autoptr(GGMLTensor) bias_output = ggml_op_add (context, weight_mul_output, repeat_bias);

  return g_steal_pointer (&bias_output);
}

GGMLTensor *
ggml_nn_layer_norm (GGMLContext *context,
                    GGMLTensor *input,
                    GGMLTensor *elementwise_weight,
                    GGMLTensor *elementwise_bias)
{
  g_autoptr(GGMLTensor) norm_output = ggml_op_norm (context, input);
  g_autoptr(GGMLTensor) repeat_elementwise_weight = ggml_op_repeat (context, elementwise_weight, norm_output);
  g_autoptr(GGMLTensor) repeat_elementwise_bias = ggml_op_repeat (context, elementwise_bias, norm_output);

  g_autoptr(GGMLTensor) elementwise_weight_output = ggml_op_mul (context, norm_output, repeat_elementwise_weight);
  g_autoptr(GGMLTensor) elementwise_bias_output = ggml_op_add (context, elementwise_weight_output, repeat_elementwise_bias);

  return g_steal_pointer (&elementwise_bias_output);
}

GGMLTensor *
ggml_nn_causal_mha_ar_layer (GGMLContext *context,
                             GGMLTensor  *input,
                             GGMLTensor  *in_attn_w,
                             GGMLTensor  *in_attn_b,
                             GGMLTensor  *out_attn_w,
                             GGMLTensor  *out_attn_b,
                             size_t       current_layer,
                             int32_t      n_embd,
                             int32_t      nhead,
                             int32_t      n_ctx,
                             int32_t      n_past,
                             int32_t      n_tokens,
                             GGMLTensor  *memory_k,
                             GGMLTensor  *memory_v,
                             GGMLTensor **out_mem_k,
                             GGMLTensor **out_mem_v)
{
  g_autoptr(GGMLTensor) proj_qkv_output = ggml_nn_linear_layer (context,
                                                                input,
                                                                in_attn_w,
                                                                in_attn_b);

  /* Chop into query, key and value head */
  g_autoptr(GGMLTensor) q_head = ggml_op_view_2d (context, proj_qkv_output, n_embd, n_tokens, 0 * n_embd);
  g_autoptr(GGMLTensor) k_head = ggml_op_view_2d (context, proj_qkv_output, n_embd, n_tokens, 1 * n_embd);
  g_autoptr(GGMLTensor) v_head = ggml_op_view_2d (context, proj_qkv_output, n_embd, n_tokens, 2 * n_embd);

  /* store into the memory tensor
   *
   * This is an optimization - basically we store the current computed keys and
   * values into a leaf-node memory and fetch from it on later iterations. This
   * means that we don't have to re-compute all the keys and values for every token
   * on each iteration, only the keys and values for the most recent token
   */
  g_autoptr(GGMLTensor) memory_view_cur_k = ggml_op_view_1d (context, memory_k, n_tokens * n_embd, n_embd * (current_layer * n_ctx + n_past));
  g_autoptr(GGMLTensor) memory_view_cur_v = ggml_op_view_1d (context, memory_v, n_tokens * n_embd, n_embd * (current_layer * n_ctx + n_past));

  /* Copy current key/value into the memory at the offset
   * and store the compute node in out_mem_k, out_mem_v */
  *out_mem_k = ggml_op_cpy (context, k_head, memory_view_cur_k);
  *out_mem_v = ggml_op_cpy (context, v_head, memory_view_cur_v);

  /* Now we continue with our computation */
  g_autoptr(GGMLTensor) q_head_contiguous_blank = ggml_context_new_tensor_3d (context, GGML_DATA_TYPE_F32, n_embd / nhead, nhead, n_tokens);
  g_autoptr(GGMLTensor) q_head_contiguous = ggml_op_cpy (context, q_head, q_head_contiguous_blank);
  g_autoptr(GGMLTensor) permuted_q_head = ggml_op_permute (context, q_head_contiguous, 0, 2, 1, 3);

  g_autoptr(GGMLTensor) memory_view_all_k = ggml_op_view_1d (context, memory_k, (n_past + n_tokens) * n_embd, current_layer * n_ctx * n_embd);
  g_autoptr(GGMLTensor) memory_view_all_v = ggml_op_view_1d (context, memory_v, (n_past + n_tokens) * n_embd, current_layer * n_ctx * n_embd);

  g_autoptr(GGMLTensor) reshaped_per_head_memory_k = ggml_op_reshape_3d (context, memory_view_all_k, n_embd / nhead, nhead, n_tokens + n_past);
  g_autoptr(GGMLTensor) permuted_per_head_memory_k = ggml_op_permute (context, reshaped_per_head_memory_k, 0, 2, 1, 3);

  g_autoptr(GGMLTensor) reshaped_per_head_memory_v = ggml_op_reshape_3d (context, memory_view_all_v, n_embd / nhead, nhead, n_tokens + n_past);
  g_autoptr(GGMLTensor) permuted_per_head_memory_v = ggml_op_permute (context, reshaped_per_head_memory_v, 1, 2, 0, 3);
  g_autoptr(GGMLTensor) permuted_per_head_memory_v_contiguous_blank = ggml_context_new_tensor_3d(context, GGML_DATA_TYPE_F32, n_tokens + n_past, n_embd / nhead, nhead);
  g_autoptr(GGMLTensor) permuted_per_head_memory_v_contiguous = ggml_op_cpy (context, permuted_per_head_memory_v, permuted_per_head_memory_v_contiguous_blank);

  /* After all that permutation, we can compute the attention matrix */
  g_autoptr(GGMLTensor) kq = ggml_op_mul_mat (context, permuted_per_head_memory_k, permuted_q_head);
  g_autoptr(GGMLTensor) scale_factor = ggml_context_new_scalar_f32 (context, 1.0 / sqrt (n_embd / nhead));
  g_autoptr(GGMLTensor) kq_scaled = ggml_op_scale_inplace (context, kq, scale_factor);
  g_autoptr(GGMLTensor) kq_masked = ggml_op_diag_mask_inf_inplace (context, kq_scaled, n_past);
  g_autoptr(GGMLTensor) kq_softmax = ggml_op_soft_max_inplace (context, kq_masked);

  /* Now that we have the attention matrix, compute A(KQ)V */
  g_autoptr(GGMLTensor) kqv = ggml_op_mul_mat (context, permuted_per_head_memory_v_contiguous, kq_softmax);
  g_autoptr(GGMLTensor) kqv_permute = ggml_op_permute (context, kqv, 0, 2, 1, 3);
  g_autoptr(GGMLTensor) kqv_permute_blank = ggml_context_new_tensor_2d (context, GGML_DATA_TYPE_F32, n_embd, n_tokens);
  g_autoptr(GGMLTensor) kqv_contiguous = ggml_op_cpy (context, kqv_permute, kqv_permute_blank);

  /* Project into output space */
  g_autoptr(GGMLTensor) output = ggml_nn_linear_layer (context, kqv_contiguous, out_attn_w, out_attn_b);

  return g_steal_pointer (&output);
}

GGMLTensor *
ggml_nn_decoder_ar_layer (GGMLContext  *context,
                          GGMLModel    *model,
                          GGMLTensor   *input,
                          size_t        i,
                          int32_t       n_embd,
                          int32_t       nhead,
                          int32_t       n_ctx,
                          int32_t       n_past,
                          int32_t       n_tokens,
                          GGMLTensor  *memory_k,
                          GGMLTensor  *memory_v,
                          GGMLTensor **out_mem_k,
                          GGMLTensor **out_mem_v)
{
  GGMLTensor *residual = input;
  g_autofree char *first_ln_g_key = g_strdup_printf ("model/h%zu/ln_1/g", i);
  g_autofree char *first_ln_b_key = g_strdup_printf ("model/h%zu/ln_1/b", i);
  g_autoptr(GGMLTensor) first_ln_output = ggml_nn_layer_norm (context,
                                                              input,
                                                              ggml_model_get (model, first_ln_g_key),
                                                              ggml_model_get (model, first_ln_b_key));

  g_autofree char *in_attn_w_key = g_strdup_printf ("model/h%zu/attn/c_attn/w", i);
  g_autofree char *in_attn_b_key = g_strdup_printf ("model/h%zu/attn/c_attn/b", i);
  g_autofree char *out_attn_w_key = g_strdup_printf ("model/h%zu/attn/c_proj/w", i);
  g_autofree char *out_attn_b_key = g_strdup_printf ("model/h%zu/attn/c_proj/b", i);
  g_autoptr(GGMLTensor) attn_output = ggml_nn_causal_mha_ar_layer (context,
                                                                   first_ln_output,
                                                                   ggml_model_get (model, in_attn_w_key),
                                                                   ggml_model_get (model, in_attn_b_key),
                                                                   ggml_model_get (model, out_attn_w_key),
                                                                   ggml_model_get (model, out_attn_b_key),
                                                                   i,
                                                                   n_embd,
                                                                   nhead,
                                                                   n_ctx,
                                                                   n_past,
                                                                   n_tokens,
                                                                   memory_k,
                                                                   memory_v,
                                                                   out_mem_k,
                                                                   out_mem_v);

  g_autoptr(GGMLTensor) attn_output_residual = ggml_op_add (context, attn_output, residual);
  GGMLTensor *residual_ff = attn_output_residual;

  g_autofree char *second_ln_g_key = g_strdup_printf ("model/h%zu/ln_2/g", i);
  g_autofree char *second_ln_b_key = g_strdup_printf ("model/h%zu/ln_2/b", i);
  g_autoptr(GGMLTensor) second_ln_output = ggml_nn_layer_norm (context,
                                                               attn_output_residual,
                                                               ggml_model_get (model, second_ln_g_key),
                                                               ggml_model_get (model, second_ln_b_key));

  g_autofree char *mlp_proj_up_w_key = g_strdup_printf ("model/h%zu/mlp/c_fc/w", i);
  g_autofree char *mlp_proj_up_b_key = g_strdup_printf ("model/h%zu/mlp/c_fc/b", i);
  g_autoptr(GGMLTensor) mlp_proj_up_output = ggml_nn_linear_layer (context,
                                                                   second_ln_output,
                                                                   ggml_model_get (model, mlp_proj_up_w_key),
                                                                   ggml_model_get (model, mlp_proj_up_b_key));
  g_autoptr(GGMLTensor) mlp_gelu_output = ggml_op_gelu (context, mlp_proj_up_output);

  g_autofree char *mlp_proj_down_w_key = g_strdup_printf ("model/h%zu/mlp/c_proj/w", i);
  g_autofree char *mlp_proj_down_b_key = g_strdup_printf ("model/h%zu/mlp/c_proj/b", i);
  g_autoptr(GGMLTensor) mlp_proj_down_output = ggml_nn_linear_layer (context,
                                                                     mlp_gelu_output,
                                                                     ggml_model_get (model, mlp_proj_down_w_key),
                                                                     ggml_model_get (model, mlp_proj_down_b_key));

  g_autoptr(GGMLTensor) mlp_residual_output = ggml_op_add (context, mlp_proj_down_output, residual_ff);

  return g_steal_pointer (&mlp_residual_output);
}

/**
 * ggml_gpt_model_forward_pass_create_memory_buffer:
 * @n_tokens: Maximum number of tokens expected to be used in this forward pass.
 *
 * Returns: (transfer full): A new #GBytes with the memory buffer needed for this
 *         forward pass.
 */
GBytes *
ggml_gpt_model_forward_pass_create_memory_buffer (size_t n_tokens)
{
  size_t estimated_size = (256 * 1024 * 1024 + (2048000 * n_tokens * 11 * 2 / 10));
  return g_bytes_new_take (g_malloc (estimated_size), estimated_size);
}

static GGMLModelDescNode *
ggml_create_gpt2_layer_model_desc (int32_t d_model,
                                   int32_t d_ff)
{
  g_autoptr(GHashTable) layer_parameters = g_hash_table_new_full (g_str_hash,
                                                                  g_str_equal,
                                                                  g_free,
                                                                  (GDestroyNotify) ggml_model_desc_node_unref);

  int64_t vector_size[] = { d_model };
  g_autoptr(GGMLModelDescNode) ln_1_g_node = ggml_model_desc_node_new_leaf (vector_size, 1, GGML_DATA_TYPE_F32);
  g_autoptr(GGMLModelDescNode) ln_1_b_node = ggml_model_desc_node_new_leaf (vector_size, 1, GGML_DATA_TYPE_F32);
  g_autoptr(GGMLModelDescNode) ln_2_g_node = ggml_model_desc_node_new_leaf (vector_size, 1, GGML_DATA_TYPE_F32);
  g_autoptr(GGMLModelDescNode) ln_2_b_node = ggml_model_desc_node_new_leaf (vector_size, 1, GGML_DATA_TYPE_F32);

  int64_t c_attn_w_size[] = { d_model, d_model * 3 };
  g_autoptr(GGMLModelDescNode) attn_c_attn_w_node = ggml_model_desc_node_new_leaf (c_attn_w_size, 2, GGML_DATA_TYPE_F16);
  g_autoptr(GGMLModelDescNode) attn_c_attn_b_node = ggml_model_desc_node_new_leaf (&c_attn_w_size[1], 1, GGML_DATA_TYPE_F32);

  int64_t c_proj_w_size[] = { d_model, d_model };
  g_autoptr(GGMLModelDescNode) attn_c_proj_w_node = ggml_model_desc_node_new_leaf (c_proj_w_size, 2, GGML_DATA_TYPE_F16);
  g_autoptr(GGMLModelDescNode) attn_c_proj_p_node = ggml_model_desc_node_new_leaf (&c_proj_w_size[1], 1, GGML_DATA_TYPE_F32);


  int64_t mlp_c_fc_w_size[] = { d_model, d_ff };
  g_autoptr(GGMLModelDescNode) mlp_c_fc_w = ggml_model_desc_node_new_leaf (mlp_c_fc_w_size, 2, GGML_DATA_TYPE_F16);
  g_autoptr(GGMLModelDescNode) mlp_c_fc_b = ggml_model_desc_node_new_leaf (&mlp_c_fc_w_size[1], 1, GGML_DATA_TYPE_F32);

  int64_t mlp_c_proj_w_size[] = { d_ff, d_model };
  g_autoptr(GGMLModelDescNode) mlp_c_proj_w = ggml_model_desc_node_new_leaf (mlp_c_proj_w_size, 2, GGML_DATA_TYPE_F16);
  g_autoptr(GGMLModelDescNode) mlp_c_proj_b = ggml_model_desc_node_new_leaf (&mlp_c_proj_w_size[1], 1, GGML_DATA_TYPE_F32);

  g_hash_table_insert (layer_parameters, g_strdup ("ln_1/g"), g_steal_pointer (&ln_1_g_node));
  g_hash_table_insert (layer_parameters, g_strdup ("ln_1/b"), g_steal_pointer (&ln_1_b_node));
  g_hash_table_insert (layer_parameters, g_strdup ("ln_2/g"), g_steal_pointer (&ln_2_g_node));
  g_hash_table_insert (layer_parameters, g_strdup ("ln_2/b"), g_steal_pointer (&ln_2_b_node));
  g_hash_table_insert (layer_parameters, g_strdup ("attn/c_attn/w"), g_steal_pointer (&attn_c_attn_w_node));
  g_hash_table_insert (layer_parameters, g_strdup ("attn/c_attn/b"), g_steal_pointer (&attn_c_attn_b_node));
  g_hash_table_insert (layer_parameters, g_strdup ("attn/c_proj/w"), g_steal_pointer (&attn_c_proj_w_node));
  g_hash_table_insert (layer_parameters, g_strdup ("attn/c_proj/b"), g_steal_pointer (&attn_c_proj_p_node));
  g_hash_table_insert (layer_parameters, g_strdup ("mlp/c_fc/w"), g_steal_pointer (&mlp_c_fc_w));
  g_hash_table_insert (layer_parameters, g_strdup ("mlp/c_fc/b"), g_steal_pointer (&mlp_c_fc_b));
  g_hash_table_insert (layer_parameters, g_strdup ("mlp/c_proj/w"), g_steal_pointer (&mlp_c_proj_w));
  g_hash_table_insert (layer_parameters, g_strdup ("mlp/c_proj/b"), g_steal_pointer (&mlp_c_proj_b));

  g_autoptr (GGMLModelDescNode) layer_node = ggml_model_desc_node_new (NULL, layer_parameters);

  return g_steal_pointer (&layer_node);
}

/**
 * ggml_create_gpt2_model_desc:
 * @n_vocab: An #int32_t with the vocab size
 * @d_model: An #int32_t with the embedding dimension
 * @d_ff: An #int32_t with the feedforward dimension
 * @n_layer: An #int32_t with the number of layers
 * @n_ctx: An #int32_t with the maximum context size
 *
 * Creates a new #GGMLModelDescNode describing the tensor layout
 * for a GPT2 model.
 *
 * Returns: (transfer full): A new #GGMLModelDescNode
 */
GGMLModelDescNode *
ggml_create_gpt2_model_desc (int32_t n_vocab,
                             int32_t d_model,
                             int32_t d_ff,
                             int32_t n_layer,
                             int32_t n_ctx)
{
  g_autoptr(GHashTable) model_parameters = g_hash_table_new_full (g_str_hash,
                                                                  g_str_equal,
                                                                  g_free,
                                                                  (GDestroyNotify) ggml_model_desc_node_unref);

  int64_t vector_size[] = { d_model };
  g_autoptr(GGMLModelDescNode) ln_f_g_node = ggml_model_desc_node_new_leaf (vector_size, 1, GGML_DATA_TYPE_F32);
  g_autoptr(GGMLModelDescNode) ln_f_b_node = ggml_model_desc_node_new_leaf (vector_size, 1, GGML_DATA_TYPE_F32);

  int64_t wte_size[] = { d_model, n_vocab };
  g_autoptr(GGMLModelDescNode) wte_node = ggml_model_desc_node_new_leaf (wte_size, 2, GGML_DATA_TYPE_F16);
  g_autoptr(GGMLModelDescNode) lm_head_node = ggml_model_desc_node_new_leaf (wte_size, 2, GGML_DATA_TYPE_F16);

  int64_t wpe_size[] = { d_model, n_ctx };
  g_autoptr(GGMLModelDescNode) wpe_node = ggml_model_desc_node_new_leaf (wpe_size, 2, GGML_DATA_TYPE_F32);

  g_hash_table_insert (model_parameters, g_strdup ("ln_f/g"), g_steal_pointer (&ln_f_g_node));
  g_hash_table_insert (model_parameters, g_strdup ("ln_f/b"), g_steal_pointer (&ln_f_b_node));
  g_hash_table_insert (model_parameters, g_strdup ("wte"), g_steal_pointer (&wte_node));
  g_hash_table_insert (model_parameters, g_strdup ("wpe"), g_steal_pointer (&wpe_node));
  g_hash_table_insert (model_parameters, g_strdup ("lm_head"), g_steal_pointer (&lm_head_node));

  for (int32_t i = 0; i < n_layer; ++i)
    {
      GGMLModelDescNode *layer_node = ggml_create_gpt2_layer_model_desc (d_model, d_ff);

      g_hash_table_insert (model_parameters, g_strdup_printf("h%d", i), layer_node);
    }

  int64_t memory_size[] = { n_layer * n_ctx * d_model };
  g_autoptr(GGMLModelDescNode) memory_k_node = ggml_model_desc_node_new_leaf (memory_size, 1, GGML_DATA_TYPE_F32);
  g_autoptr(GGMLModelDescNode) memory_v_node = ggml_model_desc_node_new_leaf (memory_size, 1, GGML_DATA_TYPE_F32);
  g_autoptr(GHashTable) memory_parameters = g_hash_table_new_full (g_str_hash,
                                                                   g_str_equal,
                                                                   g_free,
                                                                   (GDestroyNotify) ggml_model_desc_node_unref);
  g_hash_table_insert (memory_parameters, g_strdup("k"), g_steal_pointer (&memory_k_node));
  g_hash_table_insert (memory_parameters, g_strdup("v"), g_steal_pointer (&memory_v_node));

  g_autoptr(GGMLModelDescNode) memory_node = ggml_model_desc_node_new (NULL, memory_parameters);
  g_autoptr(GGMLModelDescNode) model_node = ggml_model_desc_node_new (NULL, model_parameters);

  g_autoptr(GHashTable) root_parameters = g_hash_table_new_full (g_str_hash,
                                                                 g_str_equal,
                                                                 g_free,
                                                                 (GDestroyNotify) ggml_model_desc_node_unref);
  g_hash_table_insert (root_parameters, g_strdup ("model"), g_steal_pointer (&model_node));
  g_hash_table_insert (root_parameters, g_strdup ("memory"), g_steal_pointer (&memory_node));

  g_autoptr(GGMLModelDescNode) root = ggml_model_desc_node_new (NULL, root_parameters);

  return g_steal_pointer (&root);
}

/**
 * ggml_create_gpt2_model_desc_from_hyperparameters:
 * @hyperparameters: (transfer none): A #GGMLHyperparameters
 *
 * Creates a new #GGMLModelDescNode describing the tensor layout
 * for a GPT2 model.
 *
 * Returns: (transfer full): A new #GGMLModelDescNode
 */
GGMLModelDescNode *
ggml_create_gpt2_model_desc_from_hyperparameters (GGMLHyperparameters *hyperparameters)
{
  return ggml_create_gpt2_model_desc (ggml_hyperparameters_get_int32 (hyperparameters, "n_vocab"),
                                      ggml_hyperparameters_get_int32 (hyperparameters, "n_embd"),
                                      ggml_hyperparameters_get_int32 (hyperparameters, "n_embd") * 4,
                                      ggml_hyperparameters_get_int32 (hyperparameters, "n_layer"),
                                      ggml_hyperparameters_get_int32 (hyperparameters, "n_ctx"));
}

int32_t *
read_array_from_variant (GVariant *variant, size_t *n_children)
{
  g_autoptr(GVariantIter) iter = NULL;

  g_variant_get (variant, "ai", &iter);

  g_autoptr(GArray) array = g_array_sized_new (FALSE, TRUE, sizeof (int32_t), g_variant_iter_n_children (iter));

  int32_t value;
  while (g_variant_iter_loop (iter, "i", &value))
    {
      g_array_append_vals (array, &value, 1);
    }

  return g_array_steal (array, n_children);
}

static int32_t *
arange_int32 (int32_t start, int32_t stop)
{
  g_assert (start <= stop);

  int32_t *array = g_new0 (int32_t, stop - start);

  for (int32_t i = 0; i < (stop - start); ++i)
    {
      array[i] = start + i;
    }

  return array;
}

/**
 * ggml_gpt_model_forward_pass:
 * @model: (transfer none): A #GGMLModel
 * @hyperparameters: (transfer none): A #GGMLHyperparameters
 * @inputs: (transfer none): A #GVariant with the model inputs. Should be of type "ai"
 * @input_parameters: (transfer none) (element-type utf8 int): A #GHashTable with per-pass parameters.
 *                    Should contain at least "n_past".
 * @cgraph: (transfer none): A #GGMLComputeGraph
 * @mem_buffer: (transfer none) (nullable): A #GBytes containing enough memory for this forward pass to
 *              be executed. The @mem_buffer must be sufficiently large to carry at least all the intermediate
 *              results of one forward pass. This argument can be %NULL, but if you are running the forward pass
 *              in autoregressive mode, then providing it can result in a big speedup because we can skip a lot
 *              of allocation. We also assume that nobody else is using @bytes. This function will overwrite
 *              things in @bytes, and another thread shouldn't overwrite its data.
 * @user_data: (skip): Some user data, unsued.
 * @error: A #GError out variable
 *
 * Computes the forward pass compute-graph for a GPT-decoder model from @inputs. We assume that the model
 * has keys "memory/k" and "memory/v" and they are appropriately populated given the "n_past"
 * parameter in @input_parameters. You can pass this callback directly as as #GGMLModelForwardFunc, eg
 * to ggml_model_new_from_flattened_desc.
 *
 * Note that calling this function directly does NOT run the model - it merely defines the compute
 * graph output. You need to call ggml_compute_graph_build_forward_expand on the output and then
 * ggml_compute_graph_compute, then the result will be realized in the output tensor.
 *
 * Returns: (transfer full): The output tensor node on success or %NULL with @error set on failure.
 */
GGMLTensor *
ggml_gpt_model_forward_pass (GGMLModel *model,
                             GGMLHyperparameters *hyperparameters,
                             GVariant *inputs,
                             GHashTable *input_parameters,
                             GGMLComputeGraph *cgraph,
                             GBytes *mem_buffer,
                             gpointer user_data,
                             GError **error)
{
  const int32_t n_embd = ggml_hyperparameters_get_int32 (hyperparameters, "n_embd");
  const int32_t n_layer = ggml_hyperparameters_get_int32 (hyperparameters, "n_layer");
  const int32_t n_ctx = ggml_hyperparameters_get_int32 (hyperparameters, "n_ctx");
  const int32_t nhead = ggml_hyperparameters_get_int32 (hyperparameters, "n_head");
  const int32_t n_past = GPOINTER_TO_INT (g_hash_table_lookup (input_parameters, "n_past"));

  /* We save things in the memory so that we dont have to constantly
   * recompute past keys and values that we've already computed during
   * the decoding process. */
  GGMLTensor *memory_k = ggml_model_get (model, "memory/k");
  GGMLTensor *memory_v = ggml_model_get (model, "memory/v");

  size_t n_tokens;
  g_autofree int32_t *input_tokens = read_array_from_variant (inputs, &n_tokens);
  g_autofree int32_t *positions = arange_int32 (n_past, n_past + n_tokens);

  g_autoptr(GBytes) context_mem_buffer = (
    mem_buffer != NULL ? g_bytes_ref (mem_buffer) : ggml_gpt_model_forward_pass_create_memory_buffer (n_tokens)
  );
  g_autoptr(GGMLContext) context = ggml_context_new_from_mem_buffer (context_mem_buffer);
  g_autoptr(GGMLTensor) embedding_indices = ggml_context_new_tensor_1d (context, GGML_DATA_TYPE_I32, n_tokens);
  ggml_tensor_set_data_from_int32_array (embedding_indices, input_tokens, n_tokens);

  g_autoptr(GGMLTensor) position_indices = ggml_context_new_tensor_1d (context, GGML_DATA_TYPE_I32, n_tokens);
  ggml_tensor_set_data_from_int32_array (position_indices, positions, n_tokens);

  g_autoptr(GGMLTensor) wte_rows = ggml_op_get_rows (context, ggml_model_get (model, "model/wte"), embedding_indices);
  g_autoptr(GGMLTensor) wpe_rows = ggml_op_get_rows (context, ggml_model_get (model, "model/wpe"), position_indices);

  g_autoptr(GGMLTensor) initial_inputs = ggml_op_add (context, wte_rows, wpe_rows);

  g_autoptr(GGMLTensor) residual = ggml_tensor_ref (initial_inputs);

  for (size_t i = 0; i < n_layer; ++i)
    {
      GGMLTensor *save_mem_k = NULL;
      GGMLTensor *save_mem_v = NULL;
      GGMLTensor *layer_output = ggml_nn_decoder_ar_layer (context,
                                                           model,
                                                           residual,
                                                           i,
                                                           n_embd,
                                                           nhead,
                                                           n_ctx,
                                                           n_past,
                                                           n_tokens,
                                                           memory_k,
                                                           memory_v,
                                                           &save_mem_k,
                                                           &save_mem_v);

      /* Keep the layer_output around as the next residual */
      g_clear_pointer (&residual, ggml_tensor_unref);

      /* Assigning here is fine because we the final one gets
       * owned by the autoptr and the prior ones are unref'd manually. */
      residual = layer_output;

      /* Now we need to add the memories to the compute graph
       * so that they get saved in the memory for this round */
      ggml_compute_graph_build_forward_expand (cgraph, save_mem_k);
      ggml_compute_graph_build_forward_expand (cgraph, save_mem_v);

      g_clear_pointer (&save_mem_k, ggml_tensor_unref);
      g_clear_pointer (&save_mem_v, ggml_tensor_unref);
    }

  /* Now that we have the layer outputs, we have do the final layer norm */
  g_autoptr(GGMLTensor) final_ln_output = ggml_nn_layer_norm (context,
                                                              residual,
                                                              ggml_model_get (model, "model/ln_f/g"),
                                                              ggml_model_get (model, "model/ln_f/b"));

  g_autoptr(GGMLTensor) lm_head_output = ggml_nn_linear_layer (context,
                                                               final_ln_output,
                                                               ggml_model_get (model, "model/lm_head"),
                                                               NULL);
  return g_steal_pointer (&lm_head_output);
}
