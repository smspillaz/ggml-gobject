/* global pkg, _ */
/*
 * examples/llm-writer-app/src/main.js
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
pkg.initGettext();
pkg.initFormat();
pkg.require({
    Gdk: '3.0',
    GdkPixbuf: '2.0',
    Gtk: '3.0',
    Gio: '2.0',
    GLib: '2.0',
    GObject: '2.0',
    GGML: '0'
});

const System = imports.system;
const {Gdk, GObject, Gio, GLib, Gtk, GGML} = imports.gi;

const RESOURCE_PATH = 'resource:///org/ggml-gobject/LLMWriter/Application/data';

const STATE_TEXT_EDITOR = 0;
const STATE_PREDICTING = 1;
const STATE_WAITING = 2;

const list_store_from_rows = (rows) => {
  const list_store = Gtk.ListStore.new(rows[0].map(() => GObject.TYPE_STRING));

  rows.forEach(columns => {
    const iter = list_store.append();
    columns.forEach((c, i) => {
      list_store.set_value(iter, i, c)
    });
  });

  return list_store;
};

const load_model = (model, quantization_level, cancellable, callback, progress_callback) => {
  const istream = GGML.LanguageModel.stream_from_cache(model);

  if (progress_callback) {
    istream.set_download_progress_callback(progress_callback);
  }

  const config = GGML.ModelConfig.new();

  if (quantization_level !== null)
    {
      config.set_quantization_config(
        quantization_level,
        GGML.gpt_model_quantization_regexes(),
        null
      )
    }

  GGML.LanguageModel.load_defined_from_istream_async(
    model,
    istream,
    config,
    cancellable,
    (src, res) => {
      try {
        callback(GGML.LanguageModel.load_defined_from_istream_finish(res));
      } catch (e) {
        if (e.code === Gio.IOErrorEnum.CANCELLED) {
          return;
        }
        logError(e);
      }
    }
  );
};

const COMBOBOX_ID_TO_LANGUAGE_MODEL_ENUM = Object.keys(GGML.DefinedLanguageModel).map(k => GGML.DefinedLanguageModel[k]);
const COMBOBOX_ID_TO_QUANTIZATION_LEVEL_ENUM = [
  null,
  GGML.DataType.F16,
  GGML.DataType.Q8_0,
  GGML.DataType.Q5_0,
  GGML.DataType.Q5_1,
  GGML.DataType.Q4_0,
  GGML.DataType.Q4_1,
];

class ModelLoader {
  constructor() {
    this._model_enum = null;
    this._quantization_enum = null;
    this._model = null;
    this._pending_load = null;
  }

  /**
   * with_model:
   * @model_enum: A #GGMLModelDescription
   * @cancellable: A #GCancellable
   * @callback: A callback to invoke once the model is done loading
   *
   * Does some action with a model. Also accepts a @cancellable -
   * if the action is cancelled, then @callback won't be invoked, but
   * the model will stil be downloaded if the download is in progress.
   */
  with_model(model_enum, quantization_enum, cancellable, callback, progress_callback) {
    if (this._model_enum === model_enum &&
        this._quantization_enum === quantization_enum) {
      return callback(this._model)
    }

    if (this._pending_load) {
      /* We only do the most recent callback once the model is loaded
       * and discard other ones */
      if (this._pending_load.model_enum !== model_enum ||
          this._pending_load.quantization_enum !== quantization_enum) {
        /* Cancel the existing pending load and start over again */
        this._pending_load.load_cancellable.cancel();
      } else {
        /* Don't cancel the pending load operation, but change the callback */
        this._pending_load = {
          model_enum: model_enum,
          quantization_enum: quantization_enum,
          callback: callback,
          load_cancellable: this._pending_load.load_cancellable,
          action_cancellable: cancellable
        };
        return;
      }
    }

    /* Create a pending load and load the model */
    this._pending_load = {
      model_enum: model_enum,
      quantization_enum: quantization_enum,
      callback: callback,
      load_cancellable: new Gio.Cancellable(),
      action_cancellable: cancellable
    };

    load_model(model_enum, quantization_enum, this._pending_load.load_cancellable, model => {
      const { callback, action_cancellable } = this._pending_load;

      if (action_cancellable === null || !action_cancellable.is_cancelled()) {
        this._model_enum = model_enum;
        this._quantization_enum = quantization_enum;
        this._model = model;

        System.gc();
        return callback(this._model);
      }
    }, progress_callback);
  }
}

const makeCombobox = (listOptions, callback) => {
  const combobox = Gtk.ComboBox.new_with_model(
    list_store_from_rows(listOptions)
  );
  const renderer = new Gtk.CellRendererText();
  combobox.pack_start(renderer, true);
  combobox.add_attribute(renderer, 'text', 0);
  combobox.set_active(0);
  combobox.connect('changed', callback);

  return combobox;
};

const LLMWriterAppMainWindow = GObject.registerClass({
  Template: `${RESOURCE_PATH}/main.ui`,
  Children: [
    'content-view',
    'text-view',
    'progress-bar'
  ]
}, class LLMWriterAppMainWindow extends Gtk.ApplicationWindow {
  _init(params) {
    super._init(params);

    this._model_loader = new ModelLoader();
    this._cursor = null;

    const resetProgress = () => {
      this.progress_bar.set_visible(false);
      this.progress_bar.set_text("Starting Download");
    };
    const progressCallback = (received_bytes, total_bytes) => {
      if (received_bytes === -1) {
        resetProgress();
        return;
      }

      const fraction = received_bytes / total_bytes;

      this.progress_bar.set_visible(true);
      this.progress_bar.set_fraction(fraction);
      this.progress_bar.set_text(`Downloading ${Math.trunc(fraction * 100)}%`);
    };

    const header = new Gtk.HeaderBar({
      visible: true,
      title: GLib.get_application_name(),
      show_close_button: true
    });
    this._spinner = new Gtk.Spinner({
      visible: true
    });
    const comboboxChangedCallback = () => {
      resetProgress();
      this._model_loader.with_model(
        COMBOBOX_ID_TO_LANGUAGE_MODEL_ENUM[modelCombobox.active],
        COMBOBOX_ID_TO_QUANTIZATION_LEVEL_ENUM[quantizationCombobox.active],
        null,
        () => {
          this._spinner.stop();
          this._cursor = null;
        },
        progressCallback
      );
    };
    const modelCombobox = makeCombobox([
      ['GPT2 117M'],
      ['GPT2 345M'],
      ['GPT2 774M'],
      ['GPT2 1558M'],
    ], comboboxChangedCallback);
    modelCombobox.show();
    const quantizationCombobox = makeCombobox([
      ['No quantization'],
      ['F16'],
      ['Q8_0'],
      ['Q5_0'],
      ['Q5_1'],
      ['Q4_0'],
      ['Q4_1'],
    ], comboboxChangedCallback);
    quantizationCombobox.show();

    header.pack_start(modelCombobox);
    header.pack_start(quantizationCombobox);
    header.pack_end(this._spinner);
    this.set_titlebar(header);

    this._textBufferState = STATE_TEXT_EDITOR;
    this._predictionsStartedAt = -1;
    this._cancellable = null;

    this._lastCursorOffset = this.text_view.get_buffer().cursor_position;
    const buffer = this.text_view.get_buffer();

    const removePredictedText = () => {
      const mark = buffer.get_mark("predictions-start");
      const beginIter = buffer.get_iter_at_mark(mark);
      const endIter = buffer.get_end_iter();
      this._textBufferState = STATE_TEXT_EDITOR;
      buffer.delete(beginIter, endIter);
      buffer.delete_mark(mark);
    };
    const resetState = () => {
      removePredictedText();
      this._candidateText = '';
      this.text_view.set_editable(true);
      this._spinner.stop();
      this._cursor = null;
      System.gc();
    };
    const maybeAbortPrediction = () => {
      if (this._textBufferState === STATE_PREDICTING) {
        if (this._cancellable !== null) {
          this._cancellable.cancel();
          this._cancellable = null;
        }
      }
      else if (this._textBufferState === STATE_WAITING) {
        resetState();
      }
    };
    const predictFunc = (cursor, n_tokens, prompt, textBuffer) => {
      cursor.exec_stream_async(
        n_tokens,
        2,
        this._cancellable,
        (part, is_complete_eos) => {
          if (part === prompt) {
            return;
          }

          this._candidateText += part;
          const markup = `<span foreground="gray">${GLib.markup_escape_text(part, part.length)}</span>`
          textBuffer.insert_markup(textBuffer.get_end_iter(), markup, markup.length);
          System.gc();
        },
        (src, res) => {
          try {
            cursor.exec_stream_finish(res);
          } catch (e) {
            if (e.code == Gio.IOErrorEnum.CANCELLED) {
              return;
            }
            logError(e);
            return;
          }

          this._cancellable = null;
          this._textBufferState = STATE_WAITING;
          this._spinner.stop();
        }
      );
    };

    this.text_view.connect('move-cursor', (obj, step, count, extend_selection) => {
      const currentPosition = buffer.cursor_position;
      this._lastCursorOffset = currentPosition;

      if (currentPosition > 0 &&
          currentPosition === this._lastCursorOffset &&
          count > 0 &&
          this._textBufferState === STATE_TEXT_EDITOR) {

        /* Reset state immediately if the operation is cancelled */
        this._cancellable = new Gio.Cancellable({});
        this._cancellable.connect(() => resetState());
        this._textBufferState = STATE_PREDICTING;
        this._candidateText = '';
        this._spinner.start();
        buffer.create_mark("predictions-start", buffer.get_end_iter(), true);

        if (this._cursor !== null) {
          predictFunc(this._cursor, 10, null, buffer);
        } else {
          const text = buffer.get_text(
            buffer.get_start_iter(),
            buffer.get_end_iter(),
            false
          );

          this._model_loader.with_model(
            COMBOBOX_ID_TO_LANGUAGE_MODEL_ENUM[modelCombobox.active],
            COMBOBOX_ID_TO_QUANTIZATION_LEVEL_ENUM[quantizationCombobox.active],
            this._cancellable,
            model => {
              this._cursor = model.create_completion(text, 256);
              predictFunc(this._cursor, 10, text, buffer);
            },
            progressCallback
          );
        }
      } else if (currentPosition > 0 &&
                 currentPosition === this._lastCursorOffset &&
                 count > 0 &&
                 this._textBufferState === STATE_WAITING) {
        // Delete the gray text and substitute the real text.
        removePredictedText();

        buffer.insert(buffer.get_end_iter(), this._candidateText, this._candidateText.length);
        this._candidateText = '';
        this._textBufferState = STATE_TEXT_EDITOR;
        this.text_view.set_editable(true);
      } else if (count < 0) {
        if (this._textBufferState === STATE_PREDICTING) {
          if (this._cancellable !== null) {
            this._cancellable.cancel();
            this._cancellable = null;
          }
        }

        if (this._textBufferState === STATE_WAITING) {
          resetState();
        }

        return false;
      }
    });
    this.text_view.connect('backspace', () => {
      maybeAbortPrediction();
      return false;
    });
    this.text_view.connect('insert-at-cursor', maybeAbortPrediction);
    this.text_view.connect('delete-from-cursor', maybeAbortPrediction);
    this.text_view.connect('paste-clipboard', maybeAbortPrediction);
  }

  vfunc_show() {
    super.vfunc_show();
  }
});

const LLMWriterAppApplicaiton = GObject.registerClass(class extends Gtk.Application {
  _init() {
    this._mainWindow = null;
    super._init({application_id: pkg.name});
    GLib.set_application_name(_('LLM Writer App'));
  }

  vfunc_startup() {
    super.vfunc_startup();

    const settings = Gtk.Settings.get_default();
    settings.gtk_application_prefer_dark_theme = true;
  }

  vfunc_activate() {
    if (!this._mainWindow) {
      this._mainWindow = new LLMWriterAppMainWindow({
        application: this
      }).show();
    }
  }
});

function main(argv) {
  return (new LLMWriterAppApplicaiton()).run(argv);
}
