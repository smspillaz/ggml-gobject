/*
 * ggml-gobject/ggml-language-model-sampler.c
 *
 * Library code for ggml-language-model-sampler
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

#include <ggml-gobject/ggml-language-model-sampler.h>

G_DEFINE_INTERFACE (GGMLLanguageModelSampler, ggml_language_model_sampler, G_TYPE_OBJECT)

/**
 * ggml_language_model_sampler_sample_logits_tensor:
 * @sampler: A #GGMLLanguageModelSampler
 * @logits_data: (array length=n_logits_data): A float array of logits
 * @n_logits_data: Number of elements in @logits_data
 * @shape: (array length=n_shape): Shape of @logits_data
 * @n_shape: Number of elements in @shape
 * @out_n_samples: (out): Number of elements in returned samples array
 *
 * Sample the output logits using the strategy in @sampler.
 *
 * Returns: (transfer full) (array length=out_n_samples): An array of token-ids given the
 *          the data in @tensor.
 */
size_t *
ggml_language_model_sampler_sample_logits_tensor (GGMLLanguageModelSampler *sampler,
                                                  float                    *logits_data,
                                                  size_t                    n_logits_data,
                                                  size_t                   *shape,
                                                  size_t                    n_shape,
                                                  size_t                   *out_n_samples)
{
  return GGML_LANGUAGE_MODEL_SAMPLER_GET_IFACE (sampler)->sample_logits_tensor (sampler,
                                                                                logits_data,
                                                                                n_logits_data,
                                                                                shape,
                                                                                n_shape,
                                                                                out_n_samples);
}

static void ggml_language_model_sampler_default_init (GGMLLanguageModelSamplerInterface *iface)
{
}

