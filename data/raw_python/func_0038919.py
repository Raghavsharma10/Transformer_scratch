def read_command(self):
        """
        Attempt to read the next command from the editor/server
        :return: boolean. Did we actually read a command?
        """
        # Do a non-blocking read here so the demo can keep running if there is no data
        comm = self.reader.byte(blocking=False)
        if comm is None:
            return False

        cmds = {
            SET_KEY: self.handle_set_key,
            DELETE_KEY: self.handle_delete_key,
            SET_ROW: self.handle_set_row,
            PAUSE: self.handle_pause,
            SAVE_TRACKS: self.handle_save_tracks
        }

        func = cmds.get(comm)

        if func:
            func()
        else:
            logger.error("Unknown command: %s", comm)

        return True