def create_order(dk_api, kitchen, recipe_name, variation_name, node_name=None):
        """
        returns a string.
        :param dk_api: -- api object
        :param kitchen: string
        :param recipe_name: string  -- kitchen name, string
        :param variation_name: string -- name of the recipe variation_name to be run
        :param node_name: string -- name of the single node to run
        :rtype: DKReturnCode
        """
        rc = dk_api.create_order(kitchen, recipe_name, variation_name, node_name)
        if rc.ok():
            s = 'Order ID is: %s' % rc.get_payload()
        else:
            m = rc.get_message().replace('\\n','\n')
            e = m.split('the logfile errors are:')
            if len(e) > 1:
                e2 = DKCloudCommandRunner._decompress(e[-1])
                errors = e2.split('|')
                re = e[0] + " " + 'the logfile errors are: '
                for e in errors:
                    re += '\n%s' % e
            else:
                re = m
            s = 'DKCloudCommand.create_order failed\nmessage: %s\n' % re
        rc.set_message(s)
        return rc