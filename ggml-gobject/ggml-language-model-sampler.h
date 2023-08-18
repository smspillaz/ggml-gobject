/*
 * ggml-gobject/ggml-language-model-sampler.h
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

#pragma once

#include <glib-object.h>
#include <ggml-gobject/ggml-tensor.h>

G_BEGIN_DECLS

#define GGML_TYPE_LANGUAGE_MODEL_SAMPLER ggml_language_model_sampler_get_type ()
G_DECLARE_INTERFACE (GGMLLanguageModelSampler, ggml_language_model_sampler, GGML, LANGUAGE_MODEL_SAMPLER, GObject)

struct _GGMLLanguageModelSamplerInterface
{
  GTypeInterface iface;

  size_t * (*sample_logits_tensor) (GGMLLanguageModelSampler *sampler,
                                    float                    *logits_data,
                                    size_t                    n_logits_data,
                                    size_t                   *shape,
                                    size_t                    n_shape,
                                    size_t                   *out_n_samples);
};

size_t * ggml_language_model_sampler_sample_logits_tensor (GGMLLanguageModelSampler *sampler,
                                                           float                    *logits_data,
                                                           size_t                    n_logits_data,
                                                           size_t                   *shape,
                                                           size_t                    n_shape,
                                                           size_t                   *out_n_samples);

G_END_DECLS
