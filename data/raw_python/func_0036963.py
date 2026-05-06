def cmdline_params(self,
                       distance_matrix_file_name='distance.matrix',
                       remote_folder_path=None):
        """Synthesize command line parameters

        e.g. [ ['--output-file', 'out.barcode'], ['distance_matrix.file']]

        :param distance_matrix_file_name: Name of distance matrix file
        :param remote_folder_path: Path to remote folder containing distance matrix file

        """
        parameters = []

        pm_dict = self.get_dict()
        for k, v in pm_dict.iteritems():
            parameters += ['--' + k, v]

        # distance matrix can be provided via remote folder
        if remote_folder_path is None:
            parameters += [distance_matrix_file_name]
        else:
            parameters += [remote_folder_path + distance_matrix_file_name]

        return map(str, parameters)