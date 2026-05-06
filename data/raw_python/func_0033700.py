def _input_as_path(self, data):
        """Copys the provided file to WorkingDir and returns the new filename

        data: path or filename
        """
        self._input_filename = self.getTmpFilename(
            self.WorkingDir, suffix='.fasta')
        copyfile(data, self._input_filename)
        return self._input_filename