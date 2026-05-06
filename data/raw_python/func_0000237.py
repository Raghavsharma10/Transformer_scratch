def _rewrite_paths_in_file(config_file, paths_to_replace):
        """
        Rewrite paths in config files to match convention job_xxxx/symlink
        Requires path to run_xxxx/input/config_file and a list of paths_to_replace
        """
        lines = []
        # make a copy of config
        import shutil
        shutil.copyfile(config_file, str(config_file + '_original'))
        with open(config_file) as infile:
            for line in infile:
                for old_path in paths_to_replace:
                    if old_path in line:
                        new_path = os.path.split(old_path)[-1]
                        line = line.replace(old_path, new_path)
                        logger.debug("Changed path {0} ---> {1} in file {2}".format(old_path, new_path, config_file))
                lines.append(line)
        with open(config_file, 'w') as outfile:
            for line in lines:
                outfile.write(line)