def index(self, nurest_object):
        """ Get index of the given item
            Args:
                nurest_object (bambou.NURESTObject): the NURESTObject object to verify

            Returns:
                Returns the position of the object.

            Raises:
                Raise a ValueError exception if object is not present
        """
        for index, obj in enumerate(self):
            if obj.equals(nurest_object):
                return index

        raise ValueError("%s is  not in %s" % (nurest_object, self))