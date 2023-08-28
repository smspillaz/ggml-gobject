/*
 * ggml-gobject/ggml-argmax-language-model-sampler.c
 *
 * Library code for ggml-argmax-language-model-sampler
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

#include <ggml-gobject/ggml-functional-language-model-sampler.h>

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

static size_t *
ggml_argmax_language_model_sampler_sample_logits_tensor (float    *logits_data,
                                                         size_t    n_logits_data,
                                                         size_t   *shape,
                                                         size_t    n_shape,
                                                         size_t   *out_n_samples,
                                                         gpointer  user_data)
{
  g_autofree size_t *out_tokens = g_new0 (size_t, 1);
  *out_tokens = argmax_f (logits_data, shape[0]);

  *out_n_samples = 1;
  return g_steal_pointer (&out_tokens);
}

/**
 * ggml_argmax_language_model_sampler_new:
 *
 * Returns: (transfer full): A new #GGMLLanguageModelSampler
 */
GGMLLanguageModelSampler *
ggml_argmax_language_model_sampler_new (void)
{
  return ggml_functional_language_model_sampler_new (ggml_argmax_language_model_sampler_sample_logits_tensor,
                                                     NULL,
                                                     NULL);
}

