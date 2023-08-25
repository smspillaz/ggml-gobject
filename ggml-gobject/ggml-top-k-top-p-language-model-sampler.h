/*
 * ggml-gobject/ggml-top-k-top-p-language-model-sampler.h
 *
 * Library code for ggml-top-k-top-p-language-model-sampler
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

#include <ggml-gobject/ggml-language-model-sampler.h>

G_BEGIN_DECLS

#define GGML_TYPE_TOP_K_TOP_P_LANGUAGE_MODEL_SAMPLER (ggml_top_k_top_p_language_model_sampler_get_type ())
G_DECLARE_FINAL_TYPE (GGMLTopKTopPLanguageModelSampler,
                      ggml_top_k_top_p_language_model_sampler,
                      GGML,
                      TOP_K_TOP_P_LANGUAGE_MODEL_SAMPLER,
                      GObject);

GGMLLanguageModelSampler * ggml_top_k_top_p_language_model_sampler_new (size_t top_k,
                                                                        float  top_p);

GGMLLanguageModelSampler * ggml_top_k_top_p_language_model_sampler_new_with_seed (size_t       top_k,
                                                                                  float        top_p,
                                                                                  unsigned int seed);

void
ggml_top_k_top_p_language_model_sampler_set_top_k (GGMLTopKTopPLanguageModelSampler *sampler,
                                                   unsigned int                      top_k);
unsigned int ggml_top_k_top_p_language_model_sampler_get_top_k (GGMLTopKTopPLanguageModelSampler *sampler);

void ggml_top_k_top_p_language_model_sampler_set_top_p (GGMLTopKTopPLanguageModelSampler *sampler,
                                                        float                             top_p);
float ggml_top_k_top_p_language_model_sampler_get_top_p (GGMLTopKTopPLanguageModelSampler *sampler);

void ggml_top_k_top_p_language_model_sampler_set_seed (GGMLTopKTopPLanguageModelSampler *sampler,
                                               unsigned int                      seed);

unsigned int ggml_top_k_top_p_language_model_sampler_get_seed (GGMLTopKTopPLanguageModelSampler *sampler);

G_END_DECLS

