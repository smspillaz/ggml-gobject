/*
 * ggml-gobject/ggml-gobject.h
 *
 * Header file for ggml-gobject
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

#include <ggml-gobject/ggml-cached-model.h>
#include <ggml-gobject/ggml-compute-graph.h>
#include <ggml-gobject/ggml-context.h>
#include <ggml-gobject/ggml-gpt.h>
#include <ggml-gobject/ggml-hyperparameters.h>
#include <ggml-gobject/ggml-language-model.h>
#include <ggml-gobject/ggml-language-model-sampler.h>
#include <ggml-gobject/ggml-top-k-top-p-language-model-sampler.h>
#include <ggml-gobject/ggml-model-desc.h>
#include <ggml-gobject/ggml-model.h>
#include <ggml-gobject/ggml-ops.h>
#include <ggml-gobject/ggml-quantize.h>
#include <ggml-gobject/ggml-tensor.h>
#include <ggml-gobject/ggml-token-dictionary.h>
#include <ggml-gobject/ggml-types.h>

G_BEGIN_DECLS

G_END_DECLS