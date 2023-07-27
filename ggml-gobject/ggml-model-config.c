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

#include <ggml-gobject/ggml-model-config.h>

struct _GGMLModelConfig {
  size_t ref_count;
  GGMLDataType quantization_type;
  GStrv quantization_regexes;
  GStrv skip_quantization_regexes;

  gboolean quantization_type_set : 1;
};

/**
 * ggml_model_config_new:
 *
 * Returns a new #GGMLModelConfig
 */
GGMLModelConfig *
ggml_model_config_new (void)
{
  GGMLModelConfig *config = g_new0 (GGMLModelConfig, 1);
  config->ref_count = 1;

  return config;
}

/**
 * ggml_model_config_ref: (skip)
 * @config: A #GGMLModelConfig
 *
 * Increases the ref count on @config
 *
 * Returns: (transfer full): The @config with ref count increased.
 */
GGMLModelConfig *
ggml_model_config_ref (GGMLModelConfig *config)
{
  ++config->ref_count;
  return config;
}

/**
 * ggml_model_config_unref:
 * @config: A #GGMLModelConfig
 *
 * Decreases the ref count on @config and frees the underlying config
 * if the ref count goes to zero.
 */
void
ggml_model_config_unref (GGMLModelConfig *config)
{
  if (--config->ref_count == 0)
    {
      g_clear_pointer (&config, g_free);
    }
}

/**
 * ggml_model_config_set_quantization_config:
 * @config: A #GGMLModelConfig
 * @quantization_type: A #GGMLDataType
 * @quantization_regexes: (array zero-terminated=1) (nullable): A strv of regular expressions
 *                        of weights to apply quantization to.
 * @skip_quantization_regexes: (array zero-terminated=1) (nullable): A strv of regular expressions
 *                             of weights to not quantize.
 */
void
ggml_model_config_set_quantization_config (GGMLModelConfig  *config,
                                           GGMLDataType      quantization_type,
                                           const char      **quantization_regexes,
                                           const char      **skip_quantization_regexes)
{
  config->quantization_type = quantization_type;
  config->quantization_regexes = g_strdupv ((GStrv) quantization_regexes);
  config->skip_quantization_regexes = g_strdupv ((GStrv) skip_quantization_regexes);
  config->quantization_type_set = TRUE;
}

/**
 * ggml_model_config_get_quantization_config:
 * @config: A #GGMLModelConfig
 * @out_quantization_type: (out) (nullable): A #GGMLDataType out parameter
 * @out_quantization_regexes: (out) (nullable) (array zero-terminated=1) (transfer none): The regular
 *                            expressions of weights to quantize, out-parameter.
 * @out_skip_quantization_regexes: (out) (nullable) (array zero-terminated=1) (transfer none): The regular
 *                                 expressions of weights not to quantize, out-parameter.
 *
 * Returns: %TRUE if the quantization type has been set with @quantization_type
 *          set to the internal quantization type, otherwise %FALSE
 */
gboolean
ggml_model_config_get_quantization_config (GGMLModelConfig   *config,
                                           GGMLDataType      *out_quantization_type,
                                           const char      ***out_quantization_regexes,
                                           const char      ***out_skip_quantization_regexes)
{
  if (config == NULL)
    {
      return FALSE;
    }

  if (!config->quantization_type_set)
    {
      return FALSE;
    }

  if (out_quantization_type != NULL)
    {
      *out_quantization_type = config->quantization_type;
    }

  if (out_quantization_regexes != NULL)
    {
      *out_quantization_regexes = (const char **) config->quantization_regexes;
    }

  if (out_skip_quantization_regexes != NULL)
    {
      *out_skip_quantization_regexes = (const char **) config->skip_quantization_regexes;
    }


  return TRUE;
}

G_DEFINE_BOXED_TYPE (GGMLModelConfig, ggml_model_config, ggml_model_config_ref, ggml_model_config_unref);
