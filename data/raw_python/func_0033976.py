def run(self, src_folder, requirements="requirements.txt", local_package=None):
        """Builds the file bundle.
        :param str src:
           The path to your Lambda ready project (folder must contain a valid
            config.yaml and handler module (e.g.: service.py).
        :param str local_package:
            The path to a local package with should be included in the deploy as
            well (and/or is not available on PyPi)
        """
        self.set_src_path(src_folder)

        if not os.path.isdir(self.get_src_path()):
            raise ArdyNoFileError("File {} not exist".format(self.get_src_path()))
        # Get the absolute path to the output directory and create it if it doesn't
        # already exist.
        dist_directory = 'dist'
        path_to_dist = os.path.join(self.get_src_path(), dist_directory)
        self.mkdir(path_to_dist)

        # Combine the name of the Lambda function with the current timestamp to use
        # for the output filename.
        output_filename = "{0}.zip".format(self.timestamp())

        path_to_temp = mkdtemp(prefix='aws-lambda')
        self.pip_install_to_target(path_to_temp,
                                   requirements=requirements,
                                   local_package=local_package)

        if os.path.isabs(src_folder):
            src_folder = src_folder.split(os.sep)[-1]

        self.copytree(self.get_src_path(), os.path.join(path_to_temp, src_folder))

        # Zip them together into a single file.
        # TODO: Delete temp directory created once the archive has been compiled.
        path_to_zip_file = self.create_artefact(path_to_temp, path_to_dist, output_filename)
        return path_to_zip_file