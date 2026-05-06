def patch_env(env, path, value):
        """ Set specified value to yaml path.
        Example:
        patch('application/components/child/configuration/__locator.application-id','777')
        Will change child app ID to 777
        """
        def pathGet(dictionary, path):
            for item in path.split("/"):
                dictionary = dictionary[item]
            return dictionary

        def pathSet(dictionary, path, value):
            path = path.split("/")
            key = path[-1]
            dictionary = pathGet(dictionary, "/".join(path[:-1]))
            dictionary[key] = value

        pathSet(env, path, value)
        return True