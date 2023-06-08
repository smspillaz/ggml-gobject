/*
 * tests/js/testLoadGPT2.js
 *
 * Tests for loading GPT2.
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

const { GLib, Gio, GObject, GGML } = imports.gi;

function createModelDescGPT2(n_vocab, d_model, d_ff, n_layers, n_ctx) {
  return GGML.ModelDescNode.new(
    null,
    {
      "model": GGML.ModelDescNode.new(
      null,
        {
          "ln_f/g": GGML.ModelDescNode.new_leaf([d_model], GGML.DataType.F32),
          "ln_f/b": GGML.ModelDescNode.new_leaf([d_model], GGML.DataType.F32),
          "wte": GGML.ModelDescNode.new_leaf([n_vocab, d_model], GGML.DataType.F16),
          "wpe": GGML.ModelDescNode.new_leaf([n_ctx, d_model], GGML.DataType.F32),
          "lm_head": GGML.ModelDescNode.new_leaf([d_model, n_vocab], GGML.DataType.F16),
          ...Object.fromEntries(
            [...Array(n_layers).keys()].map(
            i => [
              `h${i}`,
              GGML.ModelDescNode.new(null, {
                "ln_1/g": GGML.ModelDescNode.new_leaf([d_model], GGML.DataType.F32),
                "ln_1/b": GGML.ModelDescNode.new_leaf([d_model], GGML.DataType.F32),
                "ln_2/g": GGML.ModelDescNode.new_leaf([d_model], GGML.DataType.F32),
                "ln_2/b": GGML.ModelDescNode.new_leaf([d_model], GGML.DataType.F32),
                "attn/c_attn/w": GGML.ModelDescNode.new_leaf([d_model, 3 * d_model], GGML.DataType.F16),
                "attn/c_attn/b": GGML.ModelDescNode.new_leaf([3 * d_model], GGML.DataType.F32),
                "attn/c_proj/w": GGML.ModelDescNode.new_leaf([d_model, d_model], GGML.DataType.F16),
                "attn/c_proj/b": GGML.ModelDescNode.new_leaf([d_model], GGML.DataType.F32),
                "mlp/c_fc/w": GGML.ModelDescNode.new_leaf([d_model, d_ff], GGML.DataType.F16),
                "mlp/c_fc/b": GGML.ModelDescNode.new_leaf([d_ff], GGML.DataType.F32),
                "mlp/c_proj/w": GGML.ModelDescNode.new_leaf([d_ff, d_model], GGML.DataType.F16),
                "mlp/c_proj/b": GGML.ModelDescNode.new_leaf([d_model], GGML.DataType.F32),
              })
            ])
          )
        }
      )
    }
  );
}

describe('GGML GPT2', function() {
  it('makes the flattened dict correctly', () => {
    const model_desc = GGML.ModelDescNode.new(
      null,
      {
        "model": GGML.ModelDescNode.new(
          null,
          {
            "first": GGML.ModelDescNode.new_leaf([1], GGML.DataType.F32),
            "parent": GGML.ModelDescNode.new(
              null,
              {
                "child": GGML.ModelDescNode.new_leaf([1], GGML.DataType.F32)
              }
            )
          }
        )
      }
    );
    const flattened = model_desc.flatten();
    const flattened_keys = Object.keys(flattened);
    flattened_keys.sort();

    expect(flattened_keys).toEqual([
      "model/first",
      "model/parent/child"
    ]);
  })
  it('can construct the GPT2 model architecture', function() {
    const n_inp = 1024;
    const d_model = 768;
    const d_ff = 4 * 768;
    const n_layers = 12;
    const model_desc = createModelDescGPT2(n_inp, d_model, d_ff, n_layers);
  });
  it('can load the GPT2 weights from a bin file', function() {
    const file = Gio.File.new_for_path('../../ggml/build/models/gpt-2-117M/ggml-model.bin');
    const istream = file.read(null);

    const language_model = GGML.LanguageModel.load_from_istream(
      istream,
      (hyperparameters) => createModelDescGPT2(
        hyperparameters.get_int32('n_vocab'),
        hyperparameters.get_int32('n_embd'),
        hyperparameters.get_int32('n_embd') * 4,
        hyperparameters.get_int32('n_layer'),
        hyperparameters.get_int32('n_ctx')
      ),
      null
    );
  });
})