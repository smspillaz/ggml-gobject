/*
 * ggml-gobject/dbus/internal/ggml-client-language-model-cursor-internal.h
 *
 * Library code for ggml-language-model-cursor-client-internal
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

#include <ggml-gobject/dbus/ggml-client-language-model-cursor.h>
#include <ggml-gobject/dbus/ggml-service-dbus.h>

G_BEGIN_DECLS

GGMLClientLanguageModelCursor *
ggml_client_language_model_cursor_new (GGMLLanguageModelCompletion *proxy);

G_END_DECLS
