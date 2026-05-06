def get_compiled_serving(dk_api, kitchen, recipe_name, variation_name):
        """
        returns a string.
        :param dk_api: -- api object
        :param kitchen: string
        :param recipe_name: string  -- kitchen name, string
        :param variation_name: string -- name of the recipe variation_name to be used
        :rtype: DKReturnCode
        """
        rc = dk_api.get_compiled_serving(kitchen, recipe_name, variation_name)
        if rc.ok():
            rs = 'DKCloudCommand.get_compiled_serving succeeded %s\n' % json.dumps(rc.get_payload(), indent=4)
        else:
            m = rc.get_message()
            e = m.split('the logfile errors are:nn')
            if len(e) > 1:
                e2 = DKCloudCommandRunner._decompress(e[len(e) - 1])
                errors = e2.split('|')
                re = e[0] + " " + 'the logfile errors are: '
                for e in errors:
                    re += '\n%s' % e
            else:
                re = m
            rs = 'DKCloudCommand.get_compiled_serving failed\nmessage: %s\n' % re
        rc.set_message(rs)
        return rc