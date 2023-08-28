/*
 * ggml-gobject/ggml-compute-graph.h
 *
 * Header file for ggml-compute-graph
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
#include <ggml-gobject/ggml-compute-plan.h>
#include <ggml-gobject/ggml-context.h>
#include <ggml-gobject/ggml-tensor.h>

G_BEGIN_DECLS

typedef struct _GGMLComputeGraph GGMLComputeGraph;

#define GGML_TYPE_COMPUTE_GRAPH ggml_compute_graph_get_type ()
GType ggml_compute_graph_get_type (void);

GGMLComputeGraph * ggml_compute_graph_new (void);
GGMLComputeGraph * ggml_compute_graph_ref (GGMLComputeGraph *compute_graph);
void ggml_compute_graph_unref (GGMLComputeGraph *compute_graph);
void ggml_compute_graph_build_forward_expand (GGMLComputeGraph *compute_graph, GGMLTensor *tensor);
size_t ggml_compute_graph_get_computation_size (GGMLComputeGraph *graph,
                                                GGMLTensor       *result_tensor);
GGMLComputePlan * ggml_compute_graph_plan (GGMLComputeGraph *compute_graph, int n_threads);
gboolean ggml_compute_graph_compute (GGMLComputeGraph  *compute_graph,
                                     GGMLComputePlan   *compute_plan,
                                     GGMLContext       *context,
                                     GCancellable      *cancellable,
                                     GError           **error);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLComputeGraph, ggml_compute_graph_unref);

G_END_DECLS