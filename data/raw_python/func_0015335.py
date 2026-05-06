def emit(self, record):
        """
        Function inserts log messages to list_view
        """
        msg = record.getMessage()
        list_store = self.list_view.get_model()
        Gdk.threads_enter()
        if msg:
            # Underline URLs in the record message
            msg = replace_markup_chars(record.getMessage())
            record.msg = URL_FINDER.sub(r'<u>\1</u>', msg)
            self.parent.debug_logs['logs'].append(record)
            # During execution if level is bigger then DEBUG
            # then GUI shows the message.
            event_type = getattr(record, 'event_type', '')
            if event_type:
                if event_type == 'dep_installation_start':
                    switch_cursor(Gdk.CursorType.WATCH, self.parent.run_window)
                    list_store.append([format_entry(record)])
                if event_type == 'dep_installation_end':
                    switch_cursor(Gdk.CursorType.ARROW, self.parent.run_window)
            if not self.parent.debugging:
                # We will show only INFO messages and messages who have no dep_ event_type
                if int(record.levelno) > 10:
                    if event_type == "dep_check" or event_type == "dep_found":
                        list_store.append([format_entry(record)])
                    elif not event_type.startswith("dep_"):
                        list_store.append([format_entry(record, colorize=True)])
            if self.parent.debugging:
                if event_type != "cmd_retcode":
                    list_store.append([format_entry(record, show_level=True, colorize=True)])
        Gdk.threads_leave()