def equality(self, other):
        """Compare two objects for equality.

        @param self: first object to compare
        @param other: second object to compare

        @return: boolean result of comparison

        """
        # Compare specified attributes for equality
        cname = self.__class__.__name__
        for aname in self.attributes:
            try:
                attr1 = getattr(self, aname)
                attr2 = getattr(other, aname)
            except AttributeError as error:
                logging.debug("%s.%s: %s", cname, aname, error)
                return False
            self.log(attr1, attr2, '==', cname=cname, aname=aname)
            eql = (attr1 == attr2)
            self.log(attr1, attr2, '==', cname=cname, aname=aname, result=eql)
            if not eql:
                return False

        return True