def update_file(dk_api, kitchen, recipe_name, message, files_to_update_param):
        """
        reutrns a string.
        :param dk_api: -- api object
        :param kitchen: string
        :param recipe_name: string  -- kitchen name, string
        :param message: string message -- commit message, string
        :param files_to_update_param: string  -- file system directory where the recipe file lives
        :rtype: string
        """
        rc = DKReturnCode()
        if kitchen is None or recipe_name is None or message is None or files_to_update_param is None:
            s = 'ERROR: DKCloudCommandRunner bad input parameters'
            rc.set(rc.DK_FAIL, s)
            return rc

        # Take a simple string or an array
        if isinstance(files_to_update_param, basestring):
            files_to_update = [files_to_update_param]
        else:
            files_to_update = files_to_update_param

        msg = ''
        for file_to_update in files_to_update:
            try:
                with open(file_to_update, 'r') as f:
                    file_contents = f.read()
            except IOError as e:
                if len(msg) != 0:
                    msg += '\n'
                msg += '%s' % (str(e))
                rc.set(rc.DK_FAIL, msg)
                return rc
            except ValueError as e:
                if len(msg) != 0:
                    msg += '\n'
                msg += 'ERROR: %s' % e.message
                rc.set(rc.DK_FAIL, msg)
                return rc
            rc = dk_api.update_file(kitchen, recipe_name, message, file_to_update, file_contents)
            if not rc.ok():
                if len(msg) != 0:
                    msg += '\n'
                msg += 'DKCloudCommand.update_file for %s failed\n\tmessage: %s' % (file_to_update, rc.get_message())
                rc.set_message(msg)
                return rc
            else:
                if len(msg) != 0:
                    msg += '\n'
                msg += 'DKCloudCommand.update_file for %s succeeded' % file_to_update

        rc.set_message(msg)
        return rc