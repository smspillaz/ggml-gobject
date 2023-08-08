/*
 * ggml-gobject/ggml-compute-plan.c
 *
 * Library code for ggml-compute-plan
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

#include <ggml-gobject/ggml-compute-plan.h>
#include <ggml-gobject/internal/ggml-compute-plan-internal.h>
#include <ggml-gobject/internal/ggml-context-internal.h>
#include <ggml-gobject/internal/ggml-tensor-internal.h>

/**
 * ggml_compute_plan_ref:
 * @compute_plan: A #GGMLComputePlan
 *
 * Increase the reference count on @compute_plan
 *
 * Returns: (transfer full): The #GGMLComputePlan
 */
GGMLComputePlan *
ggml_compute_plan_ref (GGMLComputePlan *compute_plan)
{
  ++compute_plan->ref_count;
  return compute_plan;
}

/**
 * ggml_compute_plan_unref:
 * @compute_plan: A #GGMLComputePlan
 *
 * Decrease the reference count on @compute_plan . If it drops to zero
 * then @compute_plan will be freed.
 */
void
ggml_compute_plan_unref (GGMLComputePlan *compute_plan)
{
  if (--compute_plan->ref_count == 0)
    {
      g_clear_pointer (&compute_plan->cplan_work_tensor, ggml_tensor_unref);
      g_clear_pointer (&compute_plan, g_free);
    }
}

G_DEFINE_BOXED_TYPE (GGMLComputePlan,
                     ggml_compute_plan,
                     ggml_compute_plan_ref,
                     ggml_compute_plan_unref);
