def rest_equals(self, rest_object):
        """ Compare objects REST attributes

        """
        if not self.equals(rest_object):
            return False

        return self.to_dict() == rest_object.to_dict()