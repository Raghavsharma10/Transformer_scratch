def _parse_beam_file(self, file_path, run_input_dir):
        """Scan SH12A BEAM file for references to external files and return them"""
        external_files = []
        paths_to_replace = []
        with open(file_path, 'r') as beam_f:
            for line in beam_f.readlines():
                split_line = line.split()
                # line length checking to prevent IndexError
                if len(split_line) > 2 and split_line[0] == "USEBMOD":
                    logger.debug("Found reference to external file in BEAM file: {0} {1}".format(
                                 split_line[0], split_line[2]))
                    external_files.append(split_line[2])
                    paths_to_replace.append(split_line[2])
                elif len(split_line) > 1 and split_line[0] == "USECBEAM":
                    logger.debug("Found reference to external file in BEAM file: {0} {1}".format(
                                 split_line[0], split_line[1]))
                    external_files.append(split_line[1])
                    paths_to_replace.append(split_line[1])
        if paths_to_replace:
            run_dir_config_file = os.path.join(run_input_dir, os.path.split(file_path)[-1])
            logger.debug("Calling rewrite_paths method on file: {0}".format(run_dir_config_file))
            self._rewrite_paths_in_file(run_dir_config_file, paths_to_replace)
        return external_files