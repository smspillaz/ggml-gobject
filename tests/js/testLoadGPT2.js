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
          "wte": GGML.ModelDescNode.new_leaf([d_model, n_vocab], GGML.DataType.F16),
          "wpe": GGML.ModelDescNode.new_leaf([d_model, n_ctx], GGML.DataType.F32),
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
      ),
      "memory": GGML.ModelDescNode.new(
        null,
        {
          "k": GGML.ModelDescNode.new_leaf([n_layers * n_ctx * d_model], GGML.DataType.F32),
          "v": GGML.ModelDescNode.new_leaf([n_layers * n_ctx * d_model], GGML.DataType.F32),
        }
      )
    }
  );
}

const gpt2ForwardPass = (model, hyperparameters, inputs, eval_parameters, cgraph, mem_buffer) => {
  const [n_vocab, n_embd, n_ff, n_layer, n_ctx, nhead, ftype] = [
    hyperparameters.get_int32('n_vocab'),
    hyperparameters.get_int32('n_embd'),
    hyperparameters.get_int32('n_embd') * 4,
    hyperparameters.get_int32('n_layer'),
    hyperparameters.get_int32('n_ctx'),
    hyperparameters.get_int32('n_head'),
    hyperparameters.get_int32('ftype')
  ];
  const n_past = eval_parameters.n_past;
  const memory_k = model.get("memory/k");
  const memory_v = model.get("memory/v");

  /* We assume that this is enough memory for the context */
  const input_ids = inputs.deep_unpack();
  const n_tokens = input_ids.length;
  const context = GGML.Context.new_from_mem_buffer(mem_buffer);
  const embedding_indices = context.new_tensor_1d(GGML.DataType.I32, n_tokens);
  embedding_indices.set_data_from_int32_array(input_ids);
  const position_indices = context.new_tensor_1d(GGML.DataType.I32, n_tokens);
  /* Convoluted way of doing arange */
  position_indices.set_data_from_int32_array([...Array(n_tokens)].map((_, i) => i + n_past));

  const initial_input_vectors = GGML.op_add(
    context,
    GGML.op_get_rows(context, model.get("model/wte"), embedding_indices),
    GGML.op_get_rows(context, model.get("model/wpe"), position_indices)
  );

  /* Compute all the layers */
  let cur = initial_input_vectors;
  let residual = initial_input_vectors;

  for (let i = 0; i < n_layer; i++) {
    /* input layer-norm */
    cur = GGML.op_norm(context, cur);
    cur = GGML.op_add(
      context,
      GGML.op_mul(
        context,
        GGML.op_repeat(context, model.get(`model/h${i}/ln_1/g`), cur),
        cur
      ),
      GGML.op_repeat(context, model.get(`model/h${i}/ln_1/b`), cur)
    );

    /* multi-head self-attention. */
    cur = GGML.op_mul_mat(context, model.get(`model/h${i}/attn/c_attn/w`), cur);
    cur = GGML.op_add(context,
                      GGML.op_repeat(context, model.get(`model/h${i}/attn/c_attn/b`), cur),
                      cur);

    /* chop into query, key and value heads */
    const Qcur = GGML.op_view_2d(context, cur, n_embd, n_tokens, 0 * n_embd);
    const Kcur = GGML.op_view_2d(context, cur, n_embd, n_tokens, 1 * n_embd);
    const Vcur = GGML.op_view_2d(context, cur, n_embd, n_tokens, 2 * n_embd);

    /* store into the memory tensor
     *
     * This is an optimization - basically we store the current computed keys and
     * values into a leaf-node memory and fetch from it on later iterations. This
     * means that we don't have to re-compute all the keys and values for every token
     * on each iteration, only the keys and values for the most recent token
     */
    const k = GGML.op_view_1d(context, memory_k, n_tokens * n_embd, n_embd * (i * n_ctx + n_past));
    const v = GGML.op_view_1d(context, memory_v, n_tokens * n_embd, n_embd * (i * n_ctx + n_past));

    cgraph.build_forward_expand(GGML.op_cpy(context, Kcur, k));
    cgraph.build_forward_expand(GGML.op_cpy(context, Vcur, v));

    const Q = GGML.op_permute(
      context,
      GGML.op_cpy(
        context,
        Qcur,
        context.new_tensor_3d(GGML.DataType.F32, n_embd / nhead, nhead, n_tokens)
      ),
      0, 2, 1, 3
    );

    const K = GGML.op_permute(
      context,
      GGML.op_reshape_3d(
        context,
        GGML.op_view_1d(
          context,
          memory_k,
          (n_past + n_tokens) * n_embd,
          i * n_ctx * n_embd
        ),
        n_embd / nhead,
        nhead,
        n_tokens + n_past
      ),
      0, 2, 1, 3
    );

    const KQ = GGML.op_mul_mat(context, K, Q);
    const KQ_scaled = GGML.op_scale_inplace(context, KQ, context.new_scalar_f32(1.0 / Math.sqrt(n_embd / nhead)));
    const KQ_masked = GGML.op_diag_mask_inf_inplace(context, KQ_scaled, n_past);
    const KQ_soft_max = GGML.op_soft_max_inplace(context, KQ_masked);
    const V_trans = GGML.op_cpy(
      context,
      GGML.op_permute(
        context,
        GGML.op_reshape_3d(
          context,
          GGML.op_view_1d(
            context,
            memory_v,
            (n_past + n_tokens) * n_embd,
            i * n_ctx * n_embd
          ),
          n_embd / nhead,
          nhead,
          n_tokens + n_past
        ),
        1, 2, 0, 3
      ),
      context.new_tensor_3d(GGML.DataType.F32, n_tokens + n_past, n_embd / nhead, nhead)
    );

    const KQV = GGML.op_mul_mat(context, V_trans, KQ_soft_max);
    const KQV_merged = GGML.op_permute(context, KQV, 0, 2, 1, 3);

    cur = GGML.op_cpy(
      context,
      KQV_merged,
      context.new_tensor_2d(GGML.DataType.F32, n_embd, n_tokens)
    );

    /* projection */
    cur = GGML.op_mul_mat(
      context,
      model.get(`model/h${i}/attn/c_proj/w`),
      cur
    );

    cur = GGML.op_add(
      context,
      GGML.op_repeat(
        context,
        model.get(`model/h${i}/attn/c_proj/b`),
        cur
      ),
      cur
    );

    /* Add residual after projection */
    cur = GGML.op_add(context, cur, residual);

    const residualFF = cur;

    /* feedforward */

    /* feedforward norm */
    cur = GGML.op_norm(context, cur);
    cur = GGML.op_add(
      context,
      GGML.op_mul(
        context,
        GGML.op_repeat(
          context,
          model.get(`model/h${i}/ln_2/g`),
          cur
        ),
        cur
      ),
      GGML.op_repeat(context, model.get(`model/h${i}/ln_2/b`), cur)
    );

    /* feedforward fc */
    cur = GGML.op_mul_mat(
      context,
      model.get(`model/h${i}/mlp/c_fc/w`),
      cur
    );
    cur = GGML.op_add(
      context,
      GGML.op_repeat(
        context,
        model.get(`model/h${i}/mlp/c_fc/b`),
        cur
      ),
      cur
    )
    cur = GGML.op_gelu(context, cur);

    cur = GGML.op_mul_mat(
      context,
      model.get(`model/h${i}/mlp/c_proj/w`),
      cur
    );
    cur = GGML.op_add(
      context,
      GGML.op_repeat(
        context,
        model.get(`model/h${i}/mlp/c_proj/b`),
        cur
      ),
      cur
    );

    /* residual connection */
    cur = GGML.op_add(
      context,
      cur,
      residualFF
    );

    residual = cur;
  }

  /* Now we got to the end. Lets do the final norm and
    * project the outputs into logits-space */
  cur = GGML.op_norm(context, cur);
  cur = GGML.op_add(
    context,
    GGML.op_mul(
      context,
      GGML.op_repeat(
        context,
        model.get(`model/ln_f/g`),
        cur
      ),
      cur
    ),
    GGML.op_repeat(
      context,
      model.get(`model/ln_f/b`),
      cur
    )
  );

  cur = GGML.op_mul_mat(
    context,
    model.get(`model/lm_head`),
    cur
  );

  return cur;
};

describe('GGML GPT2', function() {
  it('can tokenize a simple string', function() {
    const token_dictionary = GGML.TokenDictionary.new([
      "ab",
      "bc",
      "abbcd"
    ]);
    [result, tokenized] = GGML.gpt_tokenize(token_dictionary, "abbcdabbc ab de bc");
    expect(tokenized).toEqual([2, 0, 1, 0, 1]);
  });
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
    const n_ctx = 1024;
    const model_desc = GGML.create_gpt2_model_desc(n_inp, d_model, d_ff, n_layers, n_ctx);
  });
  it('can construct the GPT2 model architecture in JS', function() {
    const n_inp = 1024;
    const d_model = 768;
    const d_ff = 4 * 768;
    const n_layers = 12;
    const n_ctx = 1024;
    const model_desc = createModelDescGPT2(n_inp, d_model, d_ff, n_layers, n_ctx);
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
      null,
      null
    );
  });
  it('can do a forward pass through some data', function() {
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
      // Bug in gjs - the "user_data" parameter is marked "skip" but it is
      // still in the function signature, so we have to curry here.
      (...args) => GGML.gpt_model_forward_pass(...args, null),
      null
    );

    expect(language_model.complete('The meaning of life is:', 7)).toEqual(
      'The meaning of life is: to live in a world of abundance'
    );
  });
  it('can do a forward pass defined in JS through some data', function() {
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
      gpt2ForwardPass,
      null
    );

    expect(language_model.complete('The meaning of life is:', 7)).toEqual(
      'The meaning of life is: to live in a world of abundance'
    );
  });
})