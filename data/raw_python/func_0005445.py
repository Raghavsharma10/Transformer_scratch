def print_live_output(self):
        '''
        Block and print the output of the command

        Raises:
            TypeError: If command is blocking
        '''
        if self.block:
            raise TypeError(NON_BLOCKING_ERROR_MESSAGE)
        else:
            while self.thread.is_alive() or self.old_output_size < len(self.output) or self.old_error_size < len(self.error):
                if self._stdout is not None and len(self.output) > self.old_output_size:
                    while self.old_output_size < len(self.output):
                        self.logger.info(self.output[self.old_output_size])
                        self.old_output_size += 1

                if self._stderr is not None and len(self.error) > self.old_error_size:
                    while self.old_error_size < len(self.error):
                        self.logger.error(self.error[self.old_error_size])
                        self.old_error_size += 1