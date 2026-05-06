def _prepare_for_submission(self, tempfolder, inputdict):
        """
        Create input files.

            :param tempfolder: aiida.common.folders.Folder subclass where
                the plugin should put all its files.
            :param inputdict: dictionary of the input nodes as they would
                be returned by get_inputs_dict
        """
        parameters, code, distance_matrix, symlink = \
                self._validate_inputs(inputdict)

        # Prepare CalcInfo to be returned to aiida
        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.remote_copy_list = []
        calcinfo.retrieve_list = parameters.output_files

        codeinfo = CodeInfo()
        codeinfo.code_uuid = code.uuid

        if distance_matrix is not None:
            calcinfo.local_copy_list = [
                [
                    distance_matrix.get_file_abs_path(),
                    distance_matrix.filename
                ],
            ]
            codeinfo.cmdline_params = parameters.cmdline_params(
                distance_matrix_file_name=distance_matrix.filename)
        else:
            calcinfo.remote_symlink_list = [symlink]
            codeinfo.cmdline_params = parameters.cmdline_params(
                remote_folder_path=self._REMOTE_FOLDER_LINK)

        calcinfo.codes_info = [codeinfo]

        return calcinfo