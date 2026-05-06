def _fetch_output_files(self, retrieved):
        """
        Checks the output folder for standard output and standard error
        files, returns their absolute paths on success.

        :param retrieved: A dictionary of retrieved nodes, as obtained from the
          parser.
        """
        from aiida.common.datastructures import calc_states
        from aiida.common.exceptions import InvalidOperation
        import os

        # check in order not to overwrite anything
        #         state = self._calc.get_state()
        #         if state != calc_states.PARSING:
        #             raise InvalidOperation("Calculation not in {} state"
        #                                    .format(calc_states.PARSING) )

        # Check that the retrieved folder is there
        try:
            out_folder = retrieved[self._calc._get_linkname_retrieved()]
        except KeyError:
            raise IOError("No retrieved folder found")

        list_of_files = out_folder.get_folder_list()

        output_path = None
        error_path = None

        if self._calc._DEFAULT_OUTPUT_FILE in list_of_files:
            output_path = os.path.join(out_folder.get_abs_path('.'),
                                       self._calc._DEFAULT_OUTPUT_FILE)
        if self._calc._DEFAULT_ERROR_FILE in list_of_files:
            error_path = os.path.join(out_folder.get_abs_path('.'),
                                      self._calc._DEFAULT_ERROR_FILE)

        return output_path, error_path