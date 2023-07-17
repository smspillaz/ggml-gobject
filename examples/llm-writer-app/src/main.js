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

    this._lastCursorOffset = this.text_view.get_buffer().cursor_position;
    this.text_view.connect('move-cursor', (obj, step, count, extend_selection) => {
      const buffer = this.text_view.get_buffer();
      const currentPosition = buffer.cursor_position;
      this._lastCursorOffset = currentPosition;

      if (currentPosition > 0 &&
          currentPosition === this._lastCursorOffset &&
          count > 0 &&
          this._languageModel !== null) {
        const text = buffer.get_text(
          buffer.get_start_iter(),
          buffer.get_end_iter(),
          false
        );

        this.text_view.set_editable(false);
        this._languageModel.complete_async(
          text,
          10,
          2,
          null,
          (src, res) => {
            const [part, is_complete, is_complete_eos] = this._languageModel.complete_finish(res);

            if (part === text) {
              return;
            }

            if (is_complete) {
              this.text_view.set_editable(true);
            }

            buffer.insert_at_cursor(part, part.length);
            System.gc();
          }
        );
      }
    });
  }

  vfunc_show() {
    let dialog = new Gtk.FileChooserDialog({
      action: Gtk.FileChooserAction.OPEN,
      select_multiple: false,
      transient_for: this,
      modal: true,
      title: 'Open'
    });

    dialog.add_button('OK', Gtk.ResponseType.OK);
    dialog.connect('response', (dialog, responseId) => {
      this._modelPath = dialog.get_filename();
      dialog.destroy();

      const file = Gio.File.new_for_path(this._modelPath);
      const istream = file.read(null);
      GGML.LanguageModel.load_defined_from_istream_async(
        GGML.DefinedLanguageModel.GPT2,
        istream,
        null,
        (src, res) => {
          this._languageModel = GGML.LanguageModel.load_defined_from_istream_finish (res);
        }
      );
    });

    super.vfunc_show();
    dialog.show();
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
