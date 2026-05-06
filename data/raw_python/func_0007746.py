def validate(self):
        """ Discover certain illegal configurations """
        if not self.editable:
            assert self.populate_from is not NotSet, u"If field (%s) is not editable, you must set populate_from" % self.name