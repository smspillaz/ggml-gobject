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
#include <ggml-gobject/ggml-closure.h>
#include <ggml-gobject/internal/ggml-closure-internal.h>

static void ggml_functional_language_model_sampler_interface_init (GGMLLanguageModelSamplerInterface *iface);

typedef struct {
  GGMLClosure *closure;
} GGMLFunctionalLanguageModelSamplerPrivate;

struct _GGMLFunctionalLanguageModelSampler {
  GObject parent_instance;
};

enum {
  PROP_0,
  PROP_CLOSURE,
  PROP_N
};

G_DEFINE_TYPE_WITH_CODE (GGMLFunctionalLanguageModelSampler,
                         ggml_functional_language_model_sampler,
                         G_TYPE_OBJECT,
                         G_ADD_PRIVATE (GGMLFunctionalLanguageModelSampler)
                         G_IMPLEMENT_INTERFACE (GGML_TYPE_LANGUAGE_MODEL_SAMPLER,
                                                ggml_functional_language_model_sampler_interface_init))

static size_t *
ggml_functional_language_model_sampler_sample_logits_tensor (GGMLLanguageModelSampler *sampler,
                                                             float                    *logits_data,
                                                             size_t                    n_logits_data,
                                                             size_t                   *shape,
                                                             size_t                    n_shape,
                                                             size_t                   *out_n_samples)
{
  GGMLFunctionalLanguageModelSampler *functional_sampler = GGML_FUNCTIONAL_LANGUAGE_MODEL_SAMPLER (sampler);
  GGMLFunctionalLanguageModelSamplerPrivate *priv = ggml_functional_language_model_sampler_get_instance_private (functional_sampler);

  return (*(GGMLLanguageModelSampleFunc) priv->closure->callback) (logits_data,
                                                                   n_logits_data,
                                                                   shape,
                                                                   n_shape,
                                                                   out_n_samples,
                                                                   priv->closure->user_data);
}

static void
ggml_functional_language_model_sampler_interface_init (GGMLLanguageModelSamplerInterface *iface)
{
  iface->sample_logits_tensor = ggml_functional_language_model_sampler_sample_logits_tensor;
}

static void
ggml_functional_language_model_sampler_dispose (GObject *object)
{
  GGMLFunctionalLanguageModelSampler *functional_sampler = GGML_FUNCTIONAL_LANGUAGE_MODEL_SAMPLER (object);
  GGMLFunctionalLanguageModelSamplerPrivate *priv = ggml_functional_language_model_sampler_get_instance_private (functional_sampler);

  g_clear_pointer (&priv->closure, ggml_closure_unref);

  return G_OBJECT_CLASS (ggml_functional_language_model_sampler_parent_class)->dispose (object);
}

static void
ggml_functional_language_model_sampler_get_property (GObject    *object,
                                                     uint32_t    prop_id,
                                                     GValue     *value,
                                                     GParamSpec *pspec)
{
  GGMLFunctionalLanguageModelSampler *functional_sampler = GGML_FUNCTIONAL_LANGUAGE_MODEL_SAMPLER (object);
  GGMLFunctionalLanguageModelSamplerPrivate *priv = ggml_functional_language_model_sampler_get_instance_private (functional_sampler);

  switch (prop_id)
    {
      case PROP_CLOSURE:
        g_value_set_boxed (value, priv->closure);
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
ggml_functional_language_model_sampler_set_property (GObject      *object,
                                                     uint32_t      prop_id,
                                                     const GValue *value,
                                                     GParamSpec   *pspec)
{
  GGMLFunctionalLanguageModelSampler *functional_sampler = GGML_FUNCTIONAL_LANGUAGE_MODEL_SAMPLER (object);
  GGMLFunctionalLanguageModelSamplerPrivate *priv = ggml_functional_language_model_sampler_get_instance_private (functional_sampler);

  switch (prop_id)
    {
      case PROP_CLOSURE:
        g_clear_pointer (&priv->closure, ggml_closure_unref);
        priv->closure = g_value_dup_boxed (value);
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
ggml_functional_language_model_sampler_class_init (GGMLFunctionalLanguageModelSamplerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = ggml_functional_language_model_sampler_dispose;
  object_class->get_property = ggml_functional_language_model_sampler_get_property;
  object_class->set_property = ggml_functional_language_model_sampler_set_property;

  g_object_class_install_property (object_class,
                                   PROP_CLOSURE,
                                   g_param_spec_boxed ("closure",
                                                       "Closure",
                                                       "Closure",
                                                       GGML_TYPE_CLOSURE,
                                                       G_PARAM_READWRITE |
                                                       G_PARAM_CONSTRUCT));
}

static void
ggml_functional_language_model_sampler_init (GGMLFunctionalLanguageModelSampler *sampler)
{
}

/**
 * ggml_functional_language_model_sampler_new:
 * @func: A #GGMLLanguageModelSampleFunc
 * @user_data: (closure func): User data for @func
 * @user_data_destroy: (destroy func): A #GDestroyNotify for @user_data
 *
 * Returns: (transfer full): A new #GGMLLanguageModelSampler with behaviour defined
 *          by @func
 */
GGMLLanguageModelSampler *
ggml_functional_language_model_sampler_new (GGMLLanguageModelSampleFunc func,
                                            gpointer                    user_data,
                                            GDestroyNotify              user_data_destroy)
{
  g_autoptr(GGMLClosure) closure = ggml_closure_new (G_CALLBACK (func),
                                                     user_data,
                                                     user_data_destroy);

  return GGML_LANGUAGE_MODEL_SAMPLER (g_object_new (GGML_TYPE_FUNCTIONAL_LANGUAGE_MODEL_SAMPLER,
                                                    "closure",
                                                    closure,
                                                    NULL));
}
