def _read_output(self, stream, callback, output_file):
        """ Read the output of the process, executed the callback and save the output.

        Args:
            stream: A file object pointing to the output stream that should be read.
            callback(callable, None): A callback function that is called for each new
                line of output.
            output_file: A file object to which the full output is written.

        Returns:
            bool: True if a line was read from the output, otherwise False.
        """
        if (callback is None and output_file is None) or stream.closed:
            return False

        line = stream.readline()
        if line:
            if callback is not None:
                callback(line.decode(),
                         self._data, self._store, self._signal, self._context)

            if output_file is not None:
                output_file.write(line)

            return True
        else:
            return False