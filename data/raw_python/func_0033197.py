def _get_result_paths(self, output_dir):
        """Return a dict of output files.
        """
        # Only include the properties file here. Add the other result
        # paths in the __call__ method, so we can catch errors if an
        # output file is not written.
        self._write_properties_file()
        properties_fp = os.path.join(self.ModelDir, self.PropertiesFile)
        result_paths = {
            'properties': ResultPath(properties_fp, IsWritten=True,)
        }
        return result_paths