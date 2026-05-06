def update_all_files(dk_api, kitchen, recipe_name, recipe_dir, message, dryrun=False):
        """
        reutrns a string.
        :param dk_api: -- api object
        :param kitchen: string
        :param recipe_name: string  -- kitchen name, string
        :param recipe_dir: string - path to the root of the directory
        :param message: string message -- commit message, string
        :rtype: DKReturnCode
        """
        rc = DKReturnCode()
        if kitchen is None or recipe_name is None or message is None:
            s = 'ERROR: DKCloudCommandRunner bad input parameters'
            rc.set(rc.DK_FAIL, s)
            return rc

        rc = dk_api.recipe_status(kitchen, recipe_name, recipe_dir)
        if not rc.ok():
            rs = 'DKCloudCommand.update_all_files failed\nmessage: %s' % rc.get_message()
            rc.set_message(rs)
            return rc

        rl = rc.get_payload()
        if (len(rl['different']) + len(rl['only_local']) + len(rl['only_remote'])) == 0:
            rs = 'DKCloudCommand.update_all_files no files changed.'
            rc.set_message(rs)
            return rc

        rc = DKCloudCommandRunner._update_changed_files(dk_api, rl['different'], kitchen, recipe_name, message, dryrun)
        if not rc.ok():
            return rc
        msg_differences = rc.get_message()

        rc = DKCloudCommandRunner._add_new_files(dk_api, rl['only_local'], kitchen, recipe_name, message, dryrun)
        if not rc.ok():
            return rc
        msg_additions = rc.get_message()

        rc = DKCloudCommandRunner._remove_deleted_files(dk_api, rl['only_remote'], kitchen, recipe_name, message,
                                                        dryrun)
        if not rc.ok():
            return rc
        msg_deletions = rc.get_message()

        msg = ''
        if len(msg_differences) > 0:
            if len(msg) > 0:
                msg += '\n'
            msg += msg_differences + '\n'
        if len(msg_additions) > 0:
            if len(msg) > 0:
                msg += '\n'
            msg += msg_additions + '\n'
        if len(msg_deletions) > 0:
            if len(msg) > 0:
                msg += '\n'
            msg += msg_deletions + '\n'
        rc.set_message(msg)
        return rc