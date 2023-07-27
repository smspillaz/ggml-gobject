/*
 * ggml-gobject/ggml-model-config.h
 *
 * Header file for ggml-model-config
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
 * You should have received a copy of the GNU Lesser General Public License
 * along with ggml-gobject; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include <glib-object.h>
#include <ggml-gobject/ggml-types.h>

G_BEGIN_DECLS

typedef struct _GGMLModelConfig GGMLModelConfig;

GGMLModelConfig * ggml_model_config_new (void);
GGMLModelConfig * ggml_model_config_ref (GGMLModelConfig *config);
void ggml_model_config_unref (GGMLModelConfig *config);

void ggml_model_config_set_quantization_config (GGMLModelConfig  *config,
                                                GGMLDataType      quantization_type,
                                                const char      **quantization_regexes,
                                                const char      **skip_quantization_regexes);
gboolean ggml_model_config_get_quantization_config (GGMLModelConfig   *config,
                                                    GGMLDataType      *out_quantization_type,
                                                    const char     ***out_quantization_regexes,
                                                    const char     ***out_skip_quantization_regexes);

#define GGML_TYPE_MODEL_CONFIG (ggml_model_config_get_type ());
GType ggml_model_config_get_type (void);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLModelConfig, ggml_model_config_unref)

G_END_DECLS
