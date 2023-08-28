/*
 * ggml-gobject/ggml-functional-language-model-sampler.h
 *
 * Library code for ggml-functional-language-model-sampler
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

#define GGML_TYPE_FUNCTIONAL_LANGUAGE_MODEL_SAMPLER (ggml_functional_language_model_sampler_get_type ())
G_DECLARE_FINAL_TYPE (GGMLFunctionalLanguageModelSampler,
                      ggml_functional_language_model_sampler,
                      GGML,
                      FUNCTIONAL_LANGUAGE_MODEL_SAMPLER,
                      GObject);

typedef size_t * (*GGMLLanguageModelSampleFunc) (float    *logits_data,
                                                 size_t    n_logits_data,
                                                 size_t   *shape,
                                                 size_t    n_shape,
                                                 size_t   *out_n_samples,
                                                 gpointer  user_data);

GGMLLanguageModelSampler * ggml_functional_language_model_sampler_new (GGMLLanguageModelSampleFunc func,
                                                                       gpointer                    user_data,
                                                                       GDestroyNotify              user_data_destroy);

G_END_DECLS
