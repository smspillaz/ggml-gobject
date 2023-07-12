/*
 * ggml-gobject/ggml-compute-graph.c
 *
 * Library code for ggml-compute-graph
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

#include <ggml-gobject/ggml-compute-graph.h>
#include <ggml-gobject/internal/ggml-context-internal.h>
#include <ggml-gobject/internal/ggml-tensor-internal.h>

struct _GGMLComputeGraph {
  struct ggml_cgraph cgraph;
  size_t ref_count;
};

/**
 * ggml_compute_graph_new:
 *
 * Returns: (transfer full): A new #GGMLComputeGraph
 */
GGMLComputeGraph *
ggml_compute_graph_new (void)
{
  GGMLComputeGraph *compute_graph = g_new0 (GGMLComputeGraph, 1);
  compute_graph->ref_count = 1;

  return compute_graph;
}

/**
 * ggml_compute_graph_ref:
 * @compute_graph: A #GGMLComputeGraph
 *
 * Increase the reference count on @compute_graph
 *
 * Returns: (transfer full): The #GGMLComputeGraph
 */
GGMLComputeGraph *
ggml_compute_graph_ref (GGMLComputeGraph *compute_graph)
{
  ++compute_graph->ref_count;
  return compute_graph;
}

/**
 * ggml_compute_graph_unref:
 * @compute_graph: A #GGMLComputeGraph
 *
 * Decrease the reference count on @compute_graph . If it drops to zero
 * then @compute_graph will be freed.
 */
void
ggml_compute_graph_unref (GGMLComputeGraph *compute_graph)
{
  if (--compute_graph->ref_count == 0)
    {
      g_clear_pointer (&compute_graph, g_free);
    }
}

/**
 * ggml_compute_graph_build_forward_expand:
 * @compute_graph: A #GGMLComputeGraph
 * @tensor: A #GGMLTensor with the end result of the computation
 *
 * Builds the internal compute graph representation based on the end result
 * tensor @tensor .
 */
void
ggml_compute_graph_build_forward_expand (GGMLComputeGraph *compute_graph, GGMLTensor *tensor)
{
  ggml_build_forward_expand (&compute_graph->cgraph, tensor->tensor);
}

/**
 * ggml_compute_graph_compute:
 * @compute_graph: A #GGMLComputeGraph
 * @context: A #GGMLContext used for the computation itself
 * @n_threads: Number of threads to use for computation. -1 means to use
 *             the number of CPUs on this machine.
 *
 * Runs the computation over the compute graph, starting from the input
 * tensors in the computation all the way to the output. After running this,
 * the tensor passed in ggml_compute_graph_build_forward_expand and all of its ancestors
 * will have some defined value.
 */
void
ggml_compute_graph_compute (GGMLComputeGraph *compute_graph,
                            GGMLContext *context,
                            int32_t      n_threads)
{
  ggml_graph_compute_with_ctx (context->ctx, &compute_graph->cgraph, n_threads);
}

G_DEFINE_BOXED_TYPE (GGMLComputeGraph,
                     ggml_compute_graph,
                     ggml_compute_graph_ref,
                     ggml_compute_graph_unref);
