def patch(self, path, value=None):
        """ Set specified value to yaml path.
        Example:
        patch('application/components/child/configuration/__locator.application-id','777')
        Will change child app ID to 777
        """
        # noinspection PyShadowingNames
        def pathGet(dictionary, path):
            for item in path.split("/"):
                dictionary = dictionary[item]
            return dictionary

        # noinspection PyShadowingNames
        def pathSet(dictionary, path, value):
            path = path.split("/")
            key = path[-1]
            dictionary = pathGet(dictionary, "/".join(path[:-1]))
            dictionary[key] = value

        # noinspection PyShadowingNames
        def pathRm(dictionary, path):
            path = path.split("/")
            key = path[-1]
            dictionary = pathGet(dictionary, "/".join(path[:-1]))
            del dictionary[key]

        src = yaml.load(self.content)
        if value:
            pathSet(src, path, value)
        else:
            pathRm(src, path)
        self._raw_content = yaml.safe_dump(src, default_flow_style=False)
        return True