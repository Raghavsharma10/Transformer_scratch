def add_file(dk_api, kitchen, recipe_name, message, api_file_key):
        """
        returns a string.
        :param dk_api: -- api object
        :param kitchen: string
        :param recipe_name: string
        :param message: string  -- commit message, string
        :param api_file_key: string  -- directory where the recipe file lives
        :rtype: DKReturnCode
        """
        rc = DKReturnCode()
        if kitchen is None or recipe_name is None or message is None or api_file_key is None:
            s = 'ERROR: DKCloudCommandRunner bad input parameters'
            rc.set(rc.DK_FAIL, s)
            return rc

        ig = DKIgnore()
        if ig.ignore(api_file_key):
            rs = 'DKCloudCommand.add_file ignoring %s' % api_file_key
            rc.set_message(rs)
            return rc

        if not os.path.exists(api_file_key):
            s = "'%s' does not exist" % api_file_key
            rc.set(rc.DK_FAIL, s)
            return rc

        try:
            with open(api_file_key, 'r') as f:
                file_contents = f.read()
        except ValueError as e:
            s = 'ERROR: %s' % e.message
            rc.set(rc.DK_FAIL, s)
            return rc
        rc = dk_api.add_file(kitchen, recipe_name, message, api_file_key, file_contents)
        if rc.ok():
            rs = 'DKCloudCommand.add_file for %s succeed' % api_file_key
        else:
            rs = 'DKCloudCommand.add_file for %s failed\nmessage: %s' % (api_file_key, rc.get_message())
        rc.set_message(rs)
        return rc