def show_progress(self, message=None):
        """If we are in a progress scope, and no log messages have been
        shown, write out another '.'"""
        if self.in_progress_hanging:
            if message is None:
                sys.stdout.write('.')
                sys.stdout.flush()
            else:
                if self.last_message:
                    padding = ' ' * max(0, len(self.last_message)-len(message))
                else:
                    padding = ''
                sys.stdout.write('\r%s%s%s%s' % (' '*self.indent, self.in_progress, message, padding))
                sys.stdout.flush()
                self.last_message = message