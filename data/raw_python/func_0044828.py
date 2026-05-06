def start(self):
        """
            start
        """
        # before printing any output to stout, we can now check this
        # variable to see if any other ProgressBar has reserved that
        # terminal.
        
        if (self.__class__.__name__ in terminal.TERMINAL_PRINT_LOOP_CLASSES):
            if not terminal.terminal_reserve(progress_obj=self):
                log.warning("tty already reserved, NOT starting the progress loop!")
                return
        
        super(Progress, self).start()
        self.show_on_exit = True