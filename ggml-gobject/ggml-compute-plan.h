/*
 * ggml-gobject/ggml-compute-plan.h
 *
 * Header file for ggml-compute-plan
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
#include <ggml-gobject/ggml-context.h>
#include <ggml-gobject/ggml-tensor.h>

G_BEGIN_DECLS

typedef struct _GGMLComputePlan GGMLComputePlan;

#define GGML_TYPE_COMPUTE_PLAN ggml_compute_plan_get_type ()
GType ggml_compute_plan_get_type (void);

GGMLComputePlan * ggml_compute_plan_ref (GGMLComputePlan *compute_plan);
void ggml_compute_plan_unref (GGMLComputePlan *compute_plan);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLComputePlan, ggml_compute_plan_unref);

G_END_DECLS