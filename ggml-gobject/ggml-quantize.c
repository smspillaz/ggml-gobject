/*
 * ggml-gobject/ggml-quantize.c
 *
 * Header file for ggml-quantize
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

#include <ggml-gobject/ggml-model-desc.h>
#include <ggml-gobject/ggml-quantize.h>

static gboolean
matches_quantize_regexes (GRegex     **quantize_keys,
                          GRegex     **skip_keys,
                          const char  *weight_path)
{
  if (skip_keys != NULL)
    {
      for (GRegex **skip_key_regex_ptr = skip_keys;
           *skip_key_regex_ptr != NULL;
           ++skip_key_regex_ptr)
        {
          if (g_regex_match (*skip_key_regex_ptr, weight_path, 0, NULL))
            {
              return FALSE;
            }
          
        }
    }

  if (quantize_keys == NULL)
    {
      return FALSE;
    }

  gboolean should_quantize = FALSE;

  for (GRegex **quantize_key_regex_ptr = quantize_keys;
       *quantize_key_regex_ptr != NULL;
       ++quantize_key_regex_ptr)
    {
      should_quantize |= g_regex_match (*quantize_key_regex_ptr, weight_path, 0, NULL);
    }

  return should_quantize;
}

static void
unref_regex_ignore_null (GRegex *regex)
{
  if (regex == NULL)
    {
      return;
    }

  return g_regex_unref (regex);
}

static gboolean
strv_to_regex_array (const char **strv,
                     GPtrArray  **out_ptr_array,
                     GError     **error)
{
  if (strv == NULL)
    {
      *out_ptr_array = NULL;
      return TRUE;
    }

  g_autoptr(GPtrArray) regex_array = g_ptr_array_new_full (g_strv_length ((GStrv) strv),
                                                           (GDestroyNotify) unref_regex_ignore_null);

  for (const char **ptr = strv; *ptr != NULL; ++ptr)
    {
      GRegex *regex = g_regex_new (*ptr, 0, 0, error);

      if (regex == NULL)
        {
          *out_ptr_array = NULL;
          return FALSE;
        }

      g_ptr_array_add (regex_array, regex);
    }

  g_ptr_array_add (regex_array, NULL);

  *out_ptr_array = g_steal_pointer (&regex_array);
  return TRUE;
}

typedef struct {
  GRegex **quantize_regex_objects;
  GRegex **skip_regex_objects;
  GGMLDataType quantize_type;
} QuantizeByRegexMapFuncData;

static GGMLModelDescLeaf *
quantize_by_regex_map_func (const char              *path,
                            const GGMLModelDescLeaf *leaf,
                            gpointer                 user_data)
{
  QuantizeByRegexMapFuncData *data = user_data;

  if (leaf->n_dim != 2 ||
      !matches_quantize_regexes (data->quantize_regex_objects, data->skip_regex_objects, path))
    {
      return ggml_model_desc_leaf_new (leaf->dimensions, leaf->n_dim, leaf->type);
    }

  return ggml_model_desc_leaf_new (leaf->dimensions, leaf->n_dim, data->quantize_type);
}

/**
 * ggml_configure_quantized_model_desc_by_regexes:
 * @model_desc: A #GGMLModelDescNode
 * @quantize_type: A #GGMLDataType
 * @quantize_regexes: (array zero-terminated=1) (nullable): A GStrv of regular expressions for weights
 *                    to quantize.
 * @skip_regexes: (array zero-terminated=1) (nullable): A #GStrv of regular expressions for weights to
 *                not quantize, if they are matched by @quantize_regexes
 * @error: A #GError out-parameter
 *
 * Returns: (transfer full): A new #GGMLModelDescNode tree with weights marked for quantization as appropriate or %NULL with
 *                           @error set on failure.
 */
GGMLModelDescNode *
ggml_configure_quantized_model_desc_by_regexes (GGMLModelDescNode  *model_desc,
                                                GGMLDataType        quantize_type,
                                                const char        **quantize_regexes,
                                                const char        **skip_regexes,
                                                GError            **error)
{
  /* We first allocate a new contex with enough memory to fit
   * the quantized model */
  g_autoptr(GPtrArray) quantize_regex_objects = NULL;
  g_autoptr(GPtrArray) skip_regex_objects = NULL;
  
  if (!strv_to_regex_array (quantize_regexes, &quantize_regex_objects, error))
    {
      return NULL;
    }

  if (!strv_to_regex_array (skip_regexes, &skip_regex_objects, error))
    {
      return NULL;
    }

  QuantizeByRegexMapFuncData data = {
    .quantize_regex_objects = quantize_regex_objects != NULL ? (GRegex **) quantize_regex_objects->pdata : NULL,
    .skip_regex_objects = skip_regex_objects != NULL ? (GRegex **) skip_regex_objects->pdata : NULL,
    .quantize_type = quantize_type
  };

  return ggml_model_desc_map (model_desc,
                              quantize_by_regex_map_func,
                              &data);
}
