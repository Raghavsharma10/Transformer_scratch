def stop(self):
        """
            trigger clean up by hand, needs to be done when not using
            context management via 'with' statement
        
            - will terminate loop process
            - show a last progress -> see the full 100% on exit
            - releases terminal reservation
        """
        super(Progress, self).stop()
        terminal.terminal_unreserve(progress_obj=self, verbose=self.verbose)

        if self.show_on_exit:
            if not isinstance(self.pipe_handler, PipeToPrint):
                myout = inMemoryBuffer()
                stdout = sys.stdout
                sys.stdout = myout
                self._show_stat()
                self.pipe_handler(myout.getvalue())
                sys.stdout = stdout
            else:
                self._show_stat()
                print()
        self.show_on_exit = False