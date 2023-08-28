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
#include <tuple>
#include <memory>

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
  g_autofree int32_t *tokens_array;
  size_t tokens_array_len;

  std::vector<int32_t> expected_tokens = {2, 0, 1, 0, 1};

  EXPECT_TRUE (ggml_gpt_tokenize (token_dictionary,
                                  "abbcdabbc ab de bc",
                                  &tokens_array,
                                  &tokens_array_len,
                                  nullptr));

  std::vector<int32_t> tokens_vector (tokens_array, tokens_array + tokens_array_len);
  EXPECT_EQ (tokens_vector, expected_tokens);
}

TEST(ModelDesc, create_gpt2_model_desc)
{
  int32_t n_inp = 1024;
  int32_t d_model = 768;
  int32_t d_ff = 4 * 768;
  int32_t n_layers = 12;
  int32_t n_ctx = 1024;
  g_autoptr(GGMLModelDescNode) model_desc = ggml_create_gpt2_model_desc(n_inp,
                                                                        d_model,
                                                                        d_ff,
                                                                        n_layers,
                                                                        n_ctx);
}

TEST(LanguageModel, load_defined_gpt2_weights)
{
  g_autoptr(GError) error = nullptr;
  g_autoptr(GGMLCachedModelIstream) istream = ggml_language_model_stream_from_cache (
    GGML_DEFINED_LANGUAGE_MODEL_GPT2P117M,
    &error
  );

  ASSERT_NE (istream, nullptr);
  ASSERT_EQ (error, nullptr);

  g_autoptr(GGMLLanguageModel) language_model = ggml_language_model_load_defined_from_istream (
    GGML_DEFINED_LANGUAGE_MODEL_GPT2P117M,
    G_INPUT_STREAM (istream),
    nullptr,
    nullptr,
    &error
  );

  ASSERT_NE (language_model, nullptr);
  ASSERT_EQ (error, nullptr);
}

typedef void (*MainLoopCallback) (GMainLoop *loop);

typedef struct {
  GMainLoop *loop;
  MainLoopCallback callback;
} WithinMainLoopData;

static WithinMainLoopData *
within_main_loop_data_new (GMainLoop *loop,
                           MainLoopCallback callback)
{
  WithinMainLoopData *data = g_new0 (WithinMainLoopData, 1);
  data->loop = g_main_loop_ref (loop);
  data->callback = callback;

  return data;
}

static void
within_main_loop_data_free (WithinMainLoopData *data)
{
  g_clear_pointer (&data->loop, g_main_loop_unref);
  g_clear_pointer (&data, g_free);
}

static void
within_main_loop (MainLoopCallback initial_callback)
{
  g_autoptr(GMainLoop) loop = g_main_loop_new (NULL, TRUE);
  WithinMainLoopData *data = within_main_loop_data_new (loop, initial_callback);
  g_idle_add ([](gpointer data) -> gboolean {
    WithinMainLoopData *within_data = (WithinMainLoopData *) data;
    within_data->callback (g_steal_pointer (&within_data->loop));
    within_main_loop_data_free (within_data);
    return G_SOURCE_REMOVE;
  }, data);
  g_main_loop_run (loop);
}

TEST(LanguageModel, load_defined_gpt2_weights_async)
{
  within_main_loop ([](GMainLoop *loop) -> void {
    g_autoptr(GError) error = nullptr;
    g_autoptr(GGMLCachedModelIstream) istream = ggml_language_model_stream_from_cache (
      GGML_DEFINED_LANGUAGE_MODEL_GPT2P117M,
      &error
    );

    ASSERT_NE (istream, nullptr);
    ASSERT_EQ (error, nullptr);

    ggml_language_model_load_defined_from_istream_async (
      GGML_DEFINED_LANGUAGE_MODEL_GPT2P117M,
      G_INPUT_STREAM (istream),
      nullptr,
      nullptr,
      [](GObject *src, GAsyncResult *result, gpointer data) -> void {
        g_autoptr(GError) error = NULL;
        g_autoptr(GMainLoop) loop = (GMainLoop *) data;
        g_autoptr(GGMLLanguageModel) language_model = ggml_language_model_load_defined_from_istream_finish (
          result,
          &error
        );

        EXPECT_EQ (error, nullptr);
        EXPECT_NE (language_model, nullptr);

        g_main_loop_quit (loop);
      },
      loop
    );
  });
}

TEST(LanguageModel, run_inference_gpt2_sync)
{
  g_autoptr(GError) error = nullptr;
  g_autoptr(GGMLCachedModelIstream) istream = ggml_language_model_stream_from_cache (
    GGML_DEFINED_LANGUAGE_MODEL_GPT2P117M,
    &error
  );

  ASSERT_NE (istream, nullptr);
  ASSERT_EQ (error, nullptr);

  g_autoptr(GGMLLanguageModel) language_model = ggml_language_model_load_defined_from_istream (
    GGML_DEFINED_LANGUAGE_MODEL_GPT2P117M,
    G_INPUT_STREAM (istream),
    nullptr,
    nullptr,
    &error
  );

  ASSERT_NE (language_model, nullptr);
  ASSERT_EQ (error, nullptr);

  g_autoptr(GGMLLanguageModelCompletionCursor) cursor = ggml_language_model_create_completion (
    language_model,
    "The meaning of life is:",
    32
  );

  gboolean is_complete_eos;
  std::string completion (ggml_language_model_completion_cursor_exec (cursor, 7, nullptr, &is_complete_eos, &error));

  ASSERT_EQ (error, nullptr);
  EXPECT_EQ (completion, "The meaning of life is: to live in a world of abundance");
}

TEST(LanguageModel, run_inference_gpt2_async)
{
  within_main_loop ([](GMainLoop *loop) -> void {
    g_autoptr(GError) error = nullptr;
    g_autoptr(GGMLCachedModelIstream) istream = ggml_language_model_stream_from_cache (
      GGML_DEFINED_LANGUAGE_MODEL_GPT2P117M,
      &error
    );

    ASSERT_NE (istream, nullptr);
    ASSERT_EQ (error, nullptr);

    g_autoptr(GGMLLanguageModel) language_model = ggml_language_model_load_defined_from_istream (
      GGML_DEFINED_LANGUAGE_MODEL_GPT2P117M,
      G_INPUT_STREAM (istream),
      nullptr,
      nullptr,
      &error
    );

    ASSERT_NE (language_model, nullptr);
    ASSERT_EQ (error, nullptr);

    g_autoptr(GGMLLanguageModelCompletionCursor) cursor = ggml_language_model_create_completion (
      language_model,
      "The meaning of life is:",
      32
    );

    typedef std::tuple <GMainLoop *, GGMLLanguageModelCompletionCursor *> ClosureData;

    ggml_language_model_completion_cursor_exec_async (
      cursor,
      7,
      nullptr,
      [](GObject *src, GAsyncResult *res, gpointer data) -> void {
        g_autoptr(GError) error = NULL;
        g_autoptr(GMainLoop) loop = NULL;
        g_autoptr(GGMLLanguageModelCompletionCursor) cursor = NULL;

        std::unique_ptr<ClosureData> closure_data ((ClosureData *) data);
        std::tie (loop, cursor) = *closure_data;

        gboolean is_complete_eos;
        std::string completion (ggml_language_model_completion_cursor_exec_finish (cursor,
                                                                                   res,
                                                                                   &is_complete_eos,
                                                                                   &error));
        EXPECT_EQ (error, nullptr);
        EXPECT_EQ (completion, "The meaning of life is: to live in a world of abundance");

        g_main_loop_quit (loop);
      },
      new ClosureData (
        g_steal_pointer (&loop),
        ggml_language_model_completion_cursor_ref (cursor)
      )
    );
  });
}

TEST(LanguageModel, run_inference_gpt2_sync_parts)
{
  g_autoptr(GError) error = nullptr;
  g_autoptr(GGMLCachedModelIstream) istream = ggml_language_model_stream_from_cache (
    GGML_DEFINED_LANGUAGE_MODEL_GPT2P117M,
    &error
  );

  ASSERT_NE (istream, nullptr);
  ASSERT_EQ (error, nullptr);

  g_autoptr(GGMLLanguageModel) language_model = ggml_language_model_load_defined_from_istream (
    GGML_DEFINED_LANGUAGE_MODEL_GPT2P117M,
    G_INPUT_STREAM (istream),
    nullptr,
    nullptr,
    &error
  );

  ASSERT_NE (language_model, nullptr);
  ASSERT_EQ (error, nullptr);

  g_autoptr(GGMLLanguageModelCompletionCursor) cursor = ggml_language_model_create_completion (
    language_model,
    "The meaning of life is:",
    32
  );

  gboolean is_complete_eos;
  std::string first_completion (ggml_language_model_completion_cursor_exec (cursor, 4, nullptr, &is_complete_eos, &error));

  ASSERT_EQ (error, nullptr);
  EXPECT_EQ (first_completion, "The meaning of life is: to live in a");

  std::string second_completion (ggml_language_model_completion_cursor_exec (cursor, 3, nullptr, &is_complete_eos, &error));

  ASSERT_EQ (error, nullptr);
  EXPECT_EQ (second_completion, " world of abundance");
}
