/*
 * ggml-gobject/ggml-stream-internal.c
 *
 * Library code for ggml-stream-internal
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

#include <ggml-gobject/internal/ggml-stream-internal.h>

gboolean
ggml_input_stream_read_exactly (GInputStream  *istream,
                                char          *buffer,
                                size_t         read_bytes,
                                GCancellable  *cancellable,
                                GError       **error)
{
  size_t bytes_read;

  if (!g_input_stream_read_all (istream,
                                buffer,
                                read_bytes,
                                &bytes_read,
                                cancellable,
                                error))
    {
      return FALSE;
    }

  if (bytes_read != read_bytes)
    {
      g_set_error (error,
                   G_IO_ERROR,
                   G_IO_ERROR_FAILED,
                   "Expected to read %zu bytes but only read %zu bytes, truncated file?",
                   read_bytes,
                   bytes_read);
      return FALSE;
    }

  return TRUE;
}
