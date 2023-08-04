/*
 * ggml-gobject/ggml-execution-memory.h
 *
 * Library code for ggml-execution-memory
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

#pragma once

#include <glib-object.h>
#include <ggml-gobject/ggml-context.h>
#include <ggml-gobject/ggml-model-desc.h>

G_BEGIN_DECLS

GType ggml_execution_memory_get_type (void);
#define GGML_TYPE_EXECUTION_MEMORY (ggml_execution_memory_get_type ())

typedef struct _GGMLExecutionMemory GGMLExecutionMemory;

GGMLExecutionMemory * ggml_execution_memory_new (size_t      execution_memory_size,
                                                 GHashTable *key_value_memory);

GGMLExecutionMemory * ggml_execution_memory_ref (GGMLExecutionMemory *memory);
void ggml_execution_memory_unref (GGMLExecutionMemory *memory);

GHashTable * ggml_execution_memory_get_key_value_memory (GGMLExecutionMemory *execution_memory);
GGMLContext * ggml_execution_memory_create_context (GGMLExecutionMemory *execution_memory);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GGMLExecutionMemory, ggml_execution_memory_unref)

G_END_DECLS
