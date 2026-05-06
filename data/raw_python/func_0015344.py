def logs_update(self):
        """
        Function updates logs.
        """
        Gdk.threads_enter()
        if not self.debugging:
            self.debugging = True
            self.debug_btn.set_label('Info logs')
        else:
            self.debugging = False
            self.debug_btn.set_label('Debug logs')
        for record in self.debug_logs['logs']:
            if self.debugging:
                # Create a new root tree element
                if getattr(record, 'event_type', '') != "cmd_retcode":
                    self.store.append([format_entry(record, show_level=True, colorize=True)])
            else:
                if int(record.levelno) > 10:
                    self.store.append([format_entry(record, colorize=True)])
        Gdk.threads_leave()