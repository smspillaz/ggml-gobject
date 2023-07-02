/*
 * ggml-gobject/ggml-types.c
 *
 * Library code for ggml-types
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

#include <ggml-types.h>

/**
 * ggml_size_of_data_type:
 * @data_type: A #GGMLDataType
 *
 * Returns: The size in bytes of this @data_type
 */
size_t
ggml_size_of_data_type (GGMLDataType data_type)
{
  return ggml_type_size ((enum ggml_type) data_type);
}