def reload(self):
        """
        Reload the program.

        :return:
            None.
        """
        # Get reload mode
        reload_mode = self._reload_mode

        # If reload mode is `exec`
        if self._reload_mode == self.RELOAD_MODE_V_EXEC:
            # Call `reload_using_exec`
            self.reload_using_exec()

        # If reload mode is `spawn_exit`
        elif self._reload_mode == self.RELOAD_MODE_V_SPAWN_EXIT:
            # Call `reload_using_spawn_exit`
            self.reload_using_spawn_exit()

        # If reload mode is `spawn_wait`
        elif self._reload_mode == self.RELOAD_MODE_V_SPAWN_WAIT:
            # Call `reload_using_spawn_wait`
            self.reload_using_spawn_wait()

        # If reload mode is none of above
        else:
            # Get error message
            error_msg = 'Invalid reload mode: {}.'.format(repr(reload_mode))

            # Raise error
            raise ValueError(error_msg)