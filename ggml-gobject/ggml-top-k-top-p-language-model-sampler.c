/*
 * ggml-gobject/ggml-top-k-top-p-language-model-sampler.c
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

#include <math.h>
#include <ggml-gobject/ggml-top-k-top-p-language-model-sampler.h>

static void ggml_top_k_top_p_language_model_sampler_interface_init (GGMLLanguageModelSamplerInterface *iface);

typedef struct {
  size_t top_k;
  float  top_p;
  size_t seed;
  gboolean seed_set;
  GRand  *rand;
} GGMLTopKTopPLanguageModelSamplerPrivate;

struct _GGMLTopKTopPLanguageModelSampler {
  GObject parent_instance;
};

enum {
  PROP_0,
  PROP_TOP_K,
  PROP_TOP_P,
  PROP_SEED,
  PROP_N
};

G_DEFINE_TYPE_WITH_CODE (GGMLTopKTopPLanguageModelSampler,
                         ggml_top_k_top_p_language_model_sampler,
                         G_TYPE_OBJECT,
                         G_ADD_PRIVATE (GGMLTopKTopPLanguageModelSampler)
                         G_IMPLEMENT_INTERFACE (GGML_TYPE_LANGUAGE_MODEL_SAMPLER,
                                                ggml_top_k_top_p_language_model_sampler_interface_init))


typedef struct {
  float  value;
  size_t idx;
} Logit;


static void
partial_sort_f32 (const float *input_buffer,
                  size_t       input_buffer_size,
                  Logit       *output_buffer,
                  size_t       output_buffer_size)
{
  g_assert (output_buffer_size >= 1);
  g_assert (output_buffer_size <= input_buffer_size);

  for (size_t i = 0; i < output_buffer_size; ++i)
    {
      output_buffer[i].value = -G_MAXFLOAT;
    }

  for (size_t i = 0; i < input_buffer_size; ++i)
    {
      if (input_buffer[i] > output_buffer[output_buffer_size - 1].value)
        {
          output_buffer[output_buffer_size - 1].value = input_buffer[i];
          output_buffer[output_buffer_size - 1].idx = i;

          for (int j = output_buffer_size - 2; j >= 0; --j)
            {
              /* Swap if output_buffer[j] < output_buffer[j + 1] -
               * eg bubble the result up */
              if (output_buffer[j + 1].value > output_buffer[j].value)
                {
                  Logit tmp = output_buffer[j];
                  output_buffer[j] = output_buffer[j + 1];
                  output_buffer[j + 1] = tmp;
                }
              else
                {
                  break;
                }
            }
        }
    }
}

static size_t *
ggml_top_k_top_p_language_model_sampler_sample_logits_tensor (GGMLLanguageModelSampler *sampler,
                                                              float                    *logits_data,
                                                              size_t                    n_logits_data,
                                                              size_t                   *shape,
                                                              size_t                    n_shape,
                                                              size_t                   *out_n_samples)
{
  GGMLTopKTopPLanguageModelSampler *top_k_top_p_sampler = GGML_TOP_K_TOP_P_LANGUAGE_MODEL_SAMPLER (sampler);
  GGMLTopKTopPLanguageModelSamplerPrivate *priv = ggml_top_k_top_p_language_model_sampler_get_instance_private (top_k_top_p_sampler);
  /* First, we need to partial-sort the logits data */
  g_autoptr(GArray) sample_array = g_array_sized_new (FALSE, FALSE, sizeof (Logit), priv->top_k);
  sample_array->len = priv->top_k;

  partial_sort_f32 (logits_data, n_logits_data, (Logit *) sample_array->data, priv->top_k);
  Logit *sample_array_logits = (Logit *) sample_array->data;

  float maxl = sample_array_logits[0].value;
  float sum = 0.0f;

  for (size_t i = 0; i < priv->top_k; ++i)
    {
      sample_array_logits[i].value = exp(sample_array_logits[i].value - maxl);
      sum += sample_array_logits[i].value;
    }

  for (size_t i = 0; i < priv->top_k; ++i)
    {
      sample_array_logits[i].value /= sum;
    }

  size_t top_p_limit = 0;
  float cumsum = 0.0f;

  for (size_t i = 0; i < priv->top_k; ++i)
    {
      float cur_value = sample_array_logits[i].value;
      sample_array_logits[i].value = cumsum;
      cumsum += cur_value;
      ++top_p_limit;

      if (cumsum >= priv->top_p)
        {
          break;
        }
    }

  for (size_t i = 0; i < top_p_limit; ++i)
    {
      sample_array_logits[i].value /= cumsum;
    }

  /* Now we uniformly sample a random number
   * and pick a logit */
  float rand_pick = g_rand_double_range (priv->rand, 0.0, 1.0);
  size_t picked = 0;

  for (; picked < top_p_limit; ++picked)
    {
      if (sample_array_logits[picked].value > rand_pick)
        {
          break;
        }
    }

  g_autofree size_t *out_tokens = g_new0 (size_t, 1);
  *out_tokens = sample_array_logits[picked - 1].idx;

  *out_n_samples = 1;
  return g_steal_pointer (&out_tokens);
}

static void
ggml_top_k_top_p_language_model_sampler_interface_init (GGMLLanguageModelSamplerInterface *iface)
{
  iface->sample_logits_tensor = ggml_top_k_top_p_language_model_sampler_sample_logits_tensor;
}

static void
ggml_top_k_top_p_language_model_sampler_get_property (GObject    *object,
                                                      guint       prop_id,
                                                      GValue     *value,
                                                      GParamSpec *pspec)
{
  GGMLTopKTopPLanguageModelSampler *sampler = GGML_TOP_K_TOP_P_LANGUAGE_MODEL_SAMPLER (object);

  switch (prop_id)
    {
      case PROP_TOP_K:
        g_value_set_uint (value, ggml_top_k_top_p_language_model_sampler_get_top_k (sampler));
        break;
      case PROP_TOP_P:
        g_value_set_float (value, ggml_top_k_top_p_language_model_sampler_get_top_p (sampler));
        break;
      case PROP_SEED:
        g_value_set_uint (value, ggml_top_k_top_p_language_model_sampler_get_seed (sampler));
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
ggml_top_k_top_p_language_model_sampler_set_property (GObject          *object,
                                                      guint             prop_id,
                                                      const GValue     *value,
                                                      GParamSpec       *pspec)
{
  GGMLTopKTopPLanguageModelSampler *sampler = GGML_TOP_K_TOP_P_LANGUAGE_MODEL_SAMPLER (object);

  switch (prop_id)
    {
      case PROP_TOP_K:
        ggml_top_k_top_p_language_model_sampler_set_top_k (sampler, g_value_get_uint (value));
        break;
      case PROP_TOP_P:
        ggml_top_k_top_p_language_model_sampler_set_top_p (sampler, g_value_get_float (value));
        break;
      case PROP_SEED:
        ggml_top_k_top_p_language_model_sampler_set_seed (sampler, g_value_get_uint (value));
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
ggml_top_k_top_p_language_model_sampler_constructed (GObject *object)
{
  GGMLTopKTopPLanguageModelSampler *sampler = GGML_TOP_K_TOP_P_LANGUAGE_MODEL_SAMPLER (object);
  GGMLTopKTopPLanguageModelSamplerPrivate *priv = ggml_top_k_top_p_language_model_sampler_get_instance_private (sampler);

  if (priv->seed_set)
    {
      priv->rand = g_rand_new_with_seed (priv->seed);
    }
  else
    {
      priv->rand = g_rand_new ();
    }
}

static void
ggml_top_k_top_p_language_model_sampler_finalize (GObject *object)
{
  GGMLTopKTopPLanguageModelSampler *sampler = GGML_TOP_K_TOP_P_LANGUAGE_MODEL_SAMPLER (object);
  GGMLTopKTopPLanguageModelSamplerPrivate *priv = ggml_top_k_top_p_language_model_sampler_get_instance_private (sampler);

  g_clear_pointer (&priv->rand, g_rand_free);
}

static void
ggml_top_k_top_p_language_model_sampler_class_init (GGMLTopKTopPLanguageModelSamplerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = ggml_top_k_top_p_language_model_sampler_get_property;
  object_class->set_property = ggml_top_k_top_p_language_model_sampler_set_property;
  object_class->finalize = ggml_top_k_top_p_language_model_sampler_finalize;
  object_class->constructed = ggml_top_k_top_p_language_model_sampler_constructed;

  g_object_class_install_property (object_class,
                                   PROP_TOP_K,
                                   g_param_spec_uint ("top-k",
                                                      "Top K",
                                                      "Top K",
                                                      1,
                                                      G_MAXUINT,
                                                      1,
                                                      G_PARAM_READWRITE |
                                                      G_PARAM_CONSTRUCT));

  g_object_class_install_property (object_class,
                                   PROP_TOP_P,
                                   g_param_spec_float ("top-p",
                                                       "Top P",
                                                       "Top P",
                                                       0.0f,
                                                       1.0f,
                                                       1.0f,
                                                       G_PARAM_READWRITE |
                                                       G_PARAM_CONSTRUCT));

  g_object_class_install_property (object_class,
                                   PROP_SEED,
                                   g_param_spec_uint ("seed",
                                                      "Seed",
                                                      "Seed",
                                                      0,
                                                      G_MAXUINT,
                                                      0,
                                                      G_PARAM_READWRITE));
}

static void
ggml_top_k_top_p_language_model_sampler_init (GGMLTopKTopPLanguageModelSampler *sampler)
{
}

/**
 * ggml_top_k_top_p_language_model_sampler_new:
 * @top_k: The top K possibilities to consider. A higher number means that you
 *         have more possibilities to select from out of all the output logits.
 * @top_p: The top P probability mass to consider. A lower value means
 *         that out of the top K tokens, only those with a cumulative
 *         probably mass of less than the value are considered. A value of 1
 *         means that we effectively consider all of the top K tokens.
 *
 * Creates a Top-k Top-p sampler. This does weighted sampling of output tokens
 * according to the distribution estimated by the model, but with some limits -
 * it will only consider the top K most likely tokens, and of those, only the ones
 * reaching the top P in probability mass. These two settings trade-off between
 * diversity of sampling versus predictability. A value of 1 for both is effectively
 * just argmax sampling. The limits allow for some diversity in the sampling but prevent
 * situations where extremely unlikely tokens are sampled by chance.
 *
 * Returns: (transfer full): A new #GGMLLanguageModelSampler with the top-k, top-p
 *          sampling methodology.
 */
GGMLLanguageModelSampler *
ggml_top_k_top_p_language_model_sampler_new (size_t top_k,
                                             float  top_p)
{
  return GGML_LANGUAGE_MODEL_SAMPLER (g_object_new (GGML_TYPE_TOP_K_TOP_P_LANGUAGE_MODEL_SAMPLER,
                                                    "top-k", top_k,
                                                    "top-p", top_p,
                                                    NULL));
}

/**
 * ggml_top_k_top_p_language_model_sampler_new_with_seed:
 * @top_k: The top K possibilities to consider. A higher number means that you
 *         have more possibilities to select from out of all the output logits.
 * @top_p: The top P probability mass to consider. A lower value means
 *         that out of the top K tokens, only those with a cumulative
 *         probably mass of less than the value are considered. A value of 1
 *         means that we effectively consider all of the top K tokens.
 *
 * Creates a Top-k Top-p sampler. See %ggml_top_k_top_p_language_model_sampler_new
 * This variant of the constructor also sets the seed value, in case you want reproducible
 * results.
 *
 * Returns: (transfer full): A new #GGMLLanguageModelSampler with the top-k, top-p
 *          sampling methodology.
 */
GGMLLanguageModelSampler * ggml_top_k_top_p_language_model_sampler_new_with_seed (size_t       top_k,
                                                                                  float        top_p,
                                                                                  unsigned int seed)
{
  return GGML_LANGUAGE_MODEL_SAMPLER (g_object_new (GGML_TYPE_TOP_K_TOP_P_LANGUAGE_MODEL_SAMPLER,
                                                    "top-k", top_k,
                                                    "top-p", top_p,
                                                    "seed", seed,
                                                    NULL));
}

void
ggml_top_k_top_p_language_model_sampler_set_top_k (GGMLTopKTopPLanguageModelSampler *sampler,
                                                   unsigned int                      top_k)
{
  GGMLTopKTopPLanguageModelSamplerPrivate *priv = ggml_top_k_top_p_language_model_sampler_get_instance_private (sampler);
  priv->top_k = top_k;
}


unsigned int
ggml_top_k_top_p_language_model_sampler_get_top_k (GGMLTopKTopPLanguageModelSampler *sampler)
{
  GGMLTopKTopPLanguageModelSamplerPrivate *priv = ggml_top_k_top_p_language_model_sampler_get_instance_private (sampler);
  return priv->top_k;
}

void
ggml_top_k_top_p_language_model_sampler_set_top_p (GGMLTopKTopPLanguageModelSampler *sampler,
                                                   float                             top_p)
{
  GGMLTopKTopPLanguageModelSamplerPrivate *priv = ggml_top_k_top_p_language_model_sampler_get_instance_private (sampler);
  priv->top_p = top_p;
}

float
ggml_top_k_top_p_language_model_sampler_get_top_p (GGMLTopKTopPLanguageModelSampler *sampler)
{
  GGMLTopKTopPLanguageModelSamplerPrivate *priv = ggml_top_k_top_p_language_model_sampler_get_instance_private (sampler);
  return priv->top_p;
}

/**
 * ggml_top_k_top_p_language_model_sampler_set_seed:
 * @sampler: A #GGMLTopKTopPLanguageModelSampler
 * @seed: The seed value
 *
 * Resets the random state of the @sampler and set the seed to @seed
 */
void
ggml_top_k_top_p_language_model_sampler_set_seed (GGMLTopKTopPLanguageModelSampler *sampler,
                                                  unsigned int                      seed)
{
  GGMLTopKTopPLanguageModelSamplerPrivate *priv = ggml_top_k_top_p_language_model_sampler_get_instance_private (sampler);

  priv->seed = seed;
  priv->seed_set = TRUE;

  if (priv->rand != NULL)
    {
      g_rand_set_seed (priv->rand, priv->seed);
    }
}

/**
 * ggml_top_k_top_p_language_model_sampler_get_seed:
 * @sampler: A #GGMLTopKTopPLanguageModelSampler
 *
 * Get the current seed value for the @sampler. If the seed was never set, then
 * the return value will be zero, but that might not be the actual seed.
 */
unsigned int
ggml_top_k_top_p_language_model_sampler_get_seed (GGMLTopKTopPLanguageModelSampler *sampler)
{
  GGMLTopKTopPLanguageModelSamplerPrivate *priv = ggml_top_k_top_p_language_model_sampler_get_instance_private (sampler);

  if (!priv->seed_set)
    {
      g_warning ("The seed was not set explcitly, so the returned value will be misleading");
    }

  return priv->seed;
}

