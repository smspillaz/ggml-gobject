/*
 * ggml-gobject/ggml-hyperparameters.h
 *
 * Header file for ggml-hyperparameters
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
#include <gio/gio.h>

G_BEGIN_DECLS

typedef struct _GGMLHyperparameters GGMLHyperparameters;

#define GGML_TYPE_HYPERPARAMETERS (ggml_hyperparameters_get_type ());
GType ggml_hyperparameters_get_type (void);

GGMLHyperparameters *ggml_hyperparameters_new (const char **ordered_keys,
                                               int *ordered_values,
                                               size_t n_ordered_values);
int32_t ggml_hyperparameters_get_int32 (GGMLHyperparameters *hyperparameters,
                                        const char *key);
GGMLHyperparameters * ggml_hyperparameters_ref (GGMLHyperparameters *hyperparameters);
void ggml_hyperparameters_unref (GGMLHyperparameters *hyperparameters);

GGMLHyperparameters * ggml_hyperparameters_load_from_istream (GInputStream *istream,
                                                              GCancellable *cancellable,
                                                              GError **error);
void ggml_hyperparameters_load_from_istream_async (GInputStream *istream,
                                                   GCancellable *cancellable,
                                                   GAsyncReadyCallback callback,
                                                   gpointer user_data);
GGMLHyperparameters * ggml_hyperparameters_load_from_istream_finish (GAsyncResult  *result,
                                                                     GError       **error);
G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLHyperparameters, ggml_hyperparameters_unref)

G_END_DECLS