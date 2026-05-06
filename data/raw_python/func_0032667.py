def generate_static(self, path):
        """
        This method generates a valid path to the public folder of the running project
        """
        if not path:
            return ""

        if path[0] == '/':
            return "%s?v=%s" % (path, self.version)

        return "%s/%s?v=%s" % (self.static, path, self.version)