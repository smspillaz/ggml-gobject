/*
 * ggml-gobject/ggml-progress-istream.h
 *
 * Library code for ggml-progress-istream
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

#include <gio/gio.h>
#include <glib-object.h>

G_BEGIN_DECLS

#define GGML_TYPE_PROGRESS_ISTREAM (ggml_progress_istream_get_type ())
G_DECLARE_FINAL_TYPE (GGMLProgressIstream, ggml_progress_istream, GGML, PROGRESS_ISTREAM, GFilterInputStream)

GGMLProgressIstream * ggml_progress_istream_new (GInputStream *base_stream,
                                                 size_t        expected_length);
void ggml_progress_istream_set_callback (GGMLProgressIstream    *progress_istream,
                                         GFileProgressCallback   callback,
                                         gpointer                user_data,
                                         GDestroyNotify          user_data_destroy);

G_END_DECLS
