def system(self):
        """The system of units used to measure an instance"""
        if self._base == 2:
            return "NIST"
        elif self._base == 10:
            return "SI"
        else:
            # I don't expect to ever encounter this logic branch, but
            # hey, it's better to have extra test coverage than
            # insufficient test coverage.
            raise ValueError("Instances mathematical base is an unsupported value: %s" % (
                str(self._base)))