/*
 * ggml-gobject/ggml-gpt.c
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

#include <ggml-gobject/ggml-execution-memory.h>

struct _GGMLExecutionMemory {
  GBytes      *execution_buffer;
  GHashTable  *key_value_memory;
  size_t       ref_count;
};

/**
 * ggml_execution_memory_new:
 * @execution_memory_size: Size in bytes of the execution memory
 * @key_value_memory: (transfer none) (nullable) (element-type utf8 GGMLTensor): A #GHashTable
 *                    of string keys and #GGMLTensor values containing the key-value memory
 *                    for this execution memory.
 *
 * Returns: (transfer full): A new #GGMLExecutionMemory
 */
GGMLExecutionMemory *
ggml_execution_memory_new (size_t      execution_memory_size,
                           GHashTable *key_value_memory)
{
  GGMLExecutionMemory *memory = g_new0 (GGMLExecutionMemory, 1);

  memory->execution_buffer = g_bytes_new_take (g_malloc (execution_memory_size), execution_memory_size);
  memory->key_value_memory = key_value_memory != NULL ? g_hash_table_ref (key_value_memory) : NULL;
  memory->ref_count = 1;

  return g_steal_pointer (&memory);
}

GGMLExecutionMemory *
ggml_execution_memory_ref (GGMLExecutionMemory *memory)
{
  ++memory->ref_count;
  return memory;
}

void
ggml_execution_memory_unref (GGMLExecutionMemory *memory)
{
  if (--memory->ref_count == 0)
    {
      g_clear_pointer (&memory->execution_buffer, g_bytes_unref);
      g_clear_pointer (&memory->key_value_memory, g_hash_table_unref);
      g_clear_pointer (&memory, g_free);
    }
}

/**
 * ggml_execution_memory_get_key_value_memory:
 * @execution_memory: A #GGMLExecutionMemory
 *
 * Returns: (transfer none) (element-type utf8 GGMLTensor): The #GHashTable for
 *          the key-value memory for this @execution_memory or %NULL if
 *          it is not set.
 */
GHashTable *
ggml_execution_memory_get_key_value_memory (GGMLExecutionMemory *execution_memory)
{
  return execution_memory->key_value_memory;
}

/**
 * ggml_execution_memory_create_context:
 * @execution_memory: A #GGMLExecutionMemory
 *
 * Returns: (transfer full): A new #GGMLContext created from this #GGMLExecutionMemory's
 *          internal buffer. It has enough space as was given on construction and can only
 *          be used for a single execution pass at a time.
 */
GGMLContext *
ggml_execution_memory_create_context (GGMLExecutionMemory *execution_memory)
{
  return ggml_context_new_from_mem_buffer (execution_memory->execution_buffer);
}

G_DEFINE_BOXED_TYPE (GGMLExecutionMemory, ggml_execution_memory, ggml_execution_memory_ref, ggml_execution_memory_unref)
