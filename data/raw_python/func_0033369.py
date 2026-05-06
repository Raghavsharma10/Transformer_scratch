def _get_result_paths(self,data):
        """ Define the output filepaths """
        output_dir = self.Parameters['--out-dir'].Value
        result = {}
        result['json'] = ResultPath(Path=join(output_dir,
                                splitext(split(self._input_filename)[-1])[0] + \
                                '.jplace'))
        return result