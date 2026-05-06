def delete_file(dk_api, kitchen, recipe_name, message, files_to_delete_param):
        """
        returns a string.
        :param dk_api: -- api object
        :param kitchen: string
        :param recipe_name: string  -- kitchen name, string
        :param message: string message -- commit message, string
        :param files_to_delete_param: path to the files to delete
        :rtype: DKReturnCode
        """
        rc = DKReturnCode()
        if kitchen is None or recipe_name is None or message is None or files_to_delete_param is None:
            s = 'ERROR: DKCloudCommandRunner bad input parameters'
            rc.set(rc.DK_FAIL, s)
            return rc

        # Take a simple string or an array
        if isinstance(files_to_delete_param, basestring):
            files_to_delete = [files_to_delete_param]
        else:
            files_to_delete = files_to_delete_param
        msg = ''
        for file_to_delete in files_to_delete:
            basename = os.path.basename(file_to_delete)
            rc = dk_api.delete_file(kitchen, recipe_name, message, file_to_delete, basename)
            if not rc.ok():
                msg += '\nDKCloudCommand.delete_file for %s failed\nmessage: %s' % (file_to_delete, rc.get_message())
                rc.set_message(msg)
                return rc
            else:
                msg += 'DKCloudCommand.delete_file for %s succeed' % file_to_delete
        rc.set_message(msg)
        return rc