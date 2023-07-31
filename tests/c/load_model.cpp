/*
 * tests/c/load_model.cpp
 *
 * Copyright (c) 2023 Sam Spilsbury
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

#include <vector>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <ggml-gobject/ggml-gobject.h>

TEST(Tokenize, simple_string)
{
  const char *dictionary_strings[] = {
    "ab",
    "bc",
    "abbcd",
    NULL
  };
  g_autoptr(GGMLTokenDictionary) token_dictionary = ggml_token_dictionary_new (dictionary_strings);
  int32_t *tokens_array;
  size_t tokens_array_len;

  std::vector<int32_t> expected_tokens = {2, 0, 1, 0, 1};

  EXPECT_TRUE (ggml_gpt_tokenize (token_dictionary,
                                  "abbcdabbc ab de bc",
                                  &tokens_array,
                                  &tokens_array_len,
                                  NULL));

  std::vector<int32_t> tokens_vector (tokens_array, tokens_array + tokens_array_len);
  EXPECT_EQ (tokens_vector, expected_tokens);
}
