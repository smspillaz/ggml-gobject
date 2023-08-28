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

#include <ggml/ggml.h>
#include <ggml/ggml-alloc.h>
#include <ggml-gobject/ggml-context.h>
#include <ggml-gobject/ggml-compute-graph.h>
#include <ggml-gobject/internal/ggml-compute-plan-internal.h>
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
 * ggml_compute_graph_get_computation_size:
 * @graph: A #GGMLComputeGraph
 * @result_tensor: A #GGMLTensor representing the result of the forward pass
 *
 * Returns the size of the allocations done on this context.
 *
 * It is an error to call this on a context that was not created using
 * %ggml_recorder_context_new
 *
 * Returns: The allocation size
 */
size_t
ggml_compute_graph_get_computation_size (GGMLComputeGraph *graph,
                                         GGMLTensor       *result_tensor)
{
  g_assert (result_tensor->owning_context->alloc != NULL);

  return ggml_allocr_alloc_graph (result_tensor->owning_context->alloc,
                                  &graph->cgraph);
}

/**
 * ggml_compute_graph_plan:
 * @compute_graph: A #GGMLComputeGraph
 * @n_threads: Number of threads to use, or -1 for default
 *
 * Returns: (transfer full): A new #GGMLComputePlan
 */
GGMLComputePlan *
ggml_compute_graph_plan (GGMLComputeGraph *compute_graph, int n_threads)
{
  GGMLComputePlan *compute_plan = g_new0 (GGMLComputePlan, 1);
  compute_plan->cplan = ggml_graph_plan (&compute_graph->cgraph, n_threads);
  compute_plan->ref_count = 1;

  size_t buffer_size = compute_plan->cplan.work_size * sizeof (int8_t) + ggml_tensor_overhead ();
  g_autoptr(GGMLContext) context = ggml_context_new (buffer_size);

  /* We have to do some manual bookkeeping of the work size here.
   * Do this first before allocating the graph, otherwise we'll end up
   * getting allocated as part of some garbage-collected tensor */
  g_autoptr(GGMLTensor) work_tensor = ggml_context_new_tensor_1d (context,
                                                                  GGML_DATA_TYPE_I8,
                                                                  compute_plan->cplan.work_size);

  compute_plan->cplan_work_tensor = g_steal_pointer (&work_tensor);
  compute_plan->cplan.work_data = compute_plan->cplan_work_tensor->tensor->data;

  return compute_plan;
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
 * @compute_plan: A #GGMLComputePlan
 * @context: A #GGMLContext
 * @cancellable: A #GCancellable
 * @error: A #GError
 *
 * Runs the computation over the compute graph, starting from the input
 * tensors in the computation all the way to the output. After running this,
 * the tensor passed in ggml_compute_graph_build_forward_expand and all of its ancestors
 * will have some defined value.
 *
 * Returns: %TRUE on success (where the end result will be in the output tensor
 *          passed to %ggml_compute_graph_build_forward_expand or %FALSE with @error
 *          set on failure)
 */
gboolean
ggml_compute_graph_compute (GGMLComputeGraph  *compute_graph,
                            GGMLComputePlan   *compute_plan,
                            GGMLContext       *context,
                            GCancellable      *cancellable,
                            GError           **error)
{
  compute_plan->cplan.abort_callback = (_Bool (*)(void*))g_cancellable_is_cancelled;
  compute_plan->cplan.abort_callback_data = cancellable;

  /* Allocate memory for computation */
  ggml_allocr_alloc_graph (context->alloc, &compute_graph->cgraph);

  int exit_status = ggml_graph_compute (&compute_graph->cgraph, &compute_plan->cplan);

  switch (exit_status)
    {
      case GGML_EXIT_ABORTED:
        g_set_error (error, G_IO_ERROR, G_IO_ERROR_CANCELLED, "Computation cancelled");
        return FALSE;
        break;
      case GGML_EXIT_SUCCESS:
        return TRUE;
    }

  g_assert_not_reached();
  return FALSE;
}

G_DEFINE_BOXED_TYPE (GGMLComputeGraph,
                     ggml_compute_graph,
                     ggml_compute_graph_ref,
                     ggml_compute_graph_unref);
