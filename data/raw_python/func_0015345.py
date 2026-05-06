def clipboard_btn_clicked(self, widget, data=None):
        """
        Function copies logs to clipboard.
        """
        _clipboard_text = []
        for record in self.debug_logs['logs']:
            if self.debugging:
                _clipboard_text.append(format_entry(record, show_level=True))
            else:
                if int(record.levelno) > 10:
                    if getattr(record, 'event_type', ''):
                        if not record.event_type.startswith("dep_"):
                            _clipboard_text.append(format_entry(record))
                    else:
                        _clipboard_text.append(format_entry(record))
        self.gui_helper.create_clipboard(_clipboard_text)