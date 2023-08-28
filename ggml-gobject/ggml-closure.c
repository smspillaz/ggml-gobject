/*
 * ggml-gobject/internal/ggml-closure.c
 *
 * Library code for ggml-closure
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

#include <glib-object.h>
#include <ggml-gobject/internal/ggml-closure-internal.h>

GGMLClosure *
ggml_closure_new (GCallback      callback,
                  gpointer       user_data,
                  GDestroyNotify user_data_destroy)
{
  GGMLClosure *closure = g_new0 (GGMLClosure, 1);
  closure->callback = callback;
  closure->user_data = user_data;
  closure->user_data_destroy = user_data_destroy;
  closure->ref_count = 1;

  return closure;
}

GGMLClosure *
ggml_closure_ref (GGMLClosure *closure)
{
  ++closure->ref_count;
  return closure;
}

void
ggml_closure_unref (GGMLClosure *closure)
{
  if (--closure->ref_count == 0)
    {
      g_clear_pointer (&closure->user_data, closure->user_data_destroy);
      g_clear_pointer (&closure, g_free);
    }
}

G_DEFINE_BOXED_TYPE (GGMLClosure,
                     ggml_closure,
                     ggml_closure_ref,
                     ggml_closure_unref);
