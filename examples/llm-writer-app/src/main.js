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

const LLMWriterAppMainWindow = GObject.registerClass({
  Template: `${RESOURCE_PATH}/main.ui`,
  Children: [
    'content-view',
    'text-view'
  ]
}, class LLMWriterAppMainWindow extends Gtk.ApplicationWindow {
  _init(params) {
    super._init(params);

    const header = new Gtk.HeaderBar({
      visible: true,
      title: GLib.get_application_name(),
      show_close_button: true
    });
    this._spinner = new Gtk.Spinner({
      visible: true
    });
    header.pack_end(this._spinner);
    this.set_titlebar(header);

    this._languageModel = null;
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

    this.text_view.connect('move-cursor', (obj, step, count, extend_selection) => {
      const currentPosition = buffer.cursor_position;
      this._lastCursorOffset = currentPosition;

      if (currentPosition > 0 &&
          currentPosition === this._lastCursorOffset &&
          count > 0 &&
          this._languageModel !== null &&
          this._textBufferState === STATE_TEXT_EDITOR) {
        const text = buffer.get_text(
          buffer.get_start_iter(),
          buffer.get_end_iter(),
          false
        );

        this.text_view.set_editable(false);
        this._cancellable = new Gio.Cancellable({});
        this._textBufferState = STATE_PREDICTING;
        this._candidateText = '';
        this._spinner.start();
        buffer.create_mark("predictions-start", buffer.get_end_iter(), true);
        this._languageModel.complete_async(
          text,
          10,
          2,
          this._cancellable,
          (src, res) => {
            let part, is_complete, is_complete_eos;
            try {
              [part, is_complete, is_complete_eos] = this._languageModel.complete_finish(res);
            } catch (e) {
              if (e.code == Gio.IOErrorEnum.CANCELLED) {
                resetState();
              }
              return;
            }

            if (part === text) {
              return;
            }

            if (is_complete) {
              this._cancellable = null;
              this._textBufferState = STATE_WAITING;
              this._spinner.stop();
            }

            this._candidateText += part;
            const markup = `<span foreground="gray">${part}</span>`
            buffer.insert_markup(buffer.get_end_iter(), markup, markup.length);
            System.gc();
          }
        );
      } else if (currentPosition > 0 &&
                 currentPosition === this._lastCursorOffset &&
                 count > 0 &&
                 this._languageModel !== null &&
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
    this._spinner.start();
    const istream = GGML.LanguageModel.stream_from_cache(GGML.DefinedLanguageModel.GPT2);
    GGML.LanguageModel.load_defined_from_istream_async(
      GGML.DefinedLanguageModel.GPT2,
      istream,
      null,
      (src, res) => {
        this._languageModel = GGML.LanguageModel.load_defined_from_istream_finish(res);
        this._spinner.stop();
      }
    );

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
