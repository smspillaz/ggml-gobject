/*
 * ggml-gobject/ggml-types.h
 *
 * Header file for ggml-types
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

#include <ggml.h>
#include <glib-object.h>

G_BEGIN_DECLS

typedef enum
{
  GGML_DATA_TYPE_F32 = GGML_TYPE_F32,
  GGML_DATA_TYPE_F16 = GGML_TYPE_F16,
  GGML_DATA_TYPE_Q4_0 = GGML_TYPE_Q4_0,
  GGML_DATA_TYPE_Q4_1 = GGML_TYPE_Q4_1,
  GGML_DATA_TYPE_Q5_0 = GGML_TYPE_Q5_0,
  GGML_DATA_TYPE_Q5_1 = GGML_TYPE_Q5_1,
  GGML_DATA_TYPE_Q8_0 = GGML_TYPE_Q8_0,
  GGML_DATA_TYPE_Q8_1 = GGML_TYPE_Q8_1,
  GGML_DATA_TYPE_I8 = GGML_TYPE_I8,
  GGML_DATA_TYPE_I16 = GGML_TYPE_I16,
  GGML_DATA_TYPE_I32 = GGML_TYPE_I32,
} GGMLDataType;

size_t ggml_size_of_data_type (GGMLDataType data_type);

G_END_DECLS