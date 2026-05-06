def cleanUp(self):
        """ Delete files that are written by CommandLineApplication from disk

            WARNING: after cleanUp() you may still have access to part of
                your result data, but you should be aware that if the file
                size exceeds the size of the buffer you will only have part
                of the file. To be safe, you should not use cleanUp() until
                you are done with the file or have copied it to a different
                location.
        """
        file_keys = self.file_keys
        for item in file_keys:
            if self[item] is not None:
                self[item].close()
                remove(self[item].name)

        # remove input handler temp files
        if hasattr(self, "_input_filename"):
            remove(self._input_filename)