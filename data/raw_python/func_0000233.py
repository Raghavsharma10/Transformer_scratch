def _parse_geo_file(self, file_path, run_input_dir):
        """Scan SH12A GEO file for references to external files (like voxelised geometry) and return them"""
        external_files = []
        paths_to_replace = []
        with open(file_path, 'r') as geo_f:
            for line in geo_f.readlines():
                split_line = line.split()
                if len(split_line) > 0 and not line.startswith("*"):
                    base_path = os.path.join(self.input_path, split_line[0])
                    if os.path.isfile(base_path + '.hed'):
                        logger.debug("Found ctx + hed files: {0}".format(base_path))
                        external_files.append(base_path + '.hed')
                        # try to find ctx file
                        if os.path.isfile(base_path + '.ctx'):
                            external_files.append(base_path + '.ctx')
                        elif os.path.isfile(base_path + '.ctx.gz'):
                            external_files.append(base_path + '.ctx.gz')
                        # replace path to match symlink location
                        paths_to_replace.append(split_line[0])
        if paths_to_replace:
            run_dir_config_file = os.path.join(run_input_dir, os.path.split(file_path)[-1])
            logger.debug("Calling rewrite_paths method on file: {0}".format(run_dir_config_file))
            self._rewrite_paths_in_file(run_dir_config_file, paths_to_replace)
        return external_files