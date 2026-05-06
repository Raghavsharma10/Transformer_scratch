def _check_format(file_path, content):
        """ check testcase format if valid
        """
        if not content:
            # testcase file content is empty
            err_msg = u"Testcase file content is empty: {}".format(file_path)
            raise p_exception.FileFormatError(err_msg)

        elif not isinstance(content, (list, dict)):
            # testcase file content does not match testcase format
            err_msg = u"Testcase file content format invalid: {}".format(file_path)
            raise p_exception.FileFormatError(err_msg)