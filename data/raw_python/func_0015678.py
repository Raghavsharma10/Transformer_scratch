def pack_pointer(self, name):
        """Returns a pointer containing the value.

        This only works for int32/uint32/utf-8..
        """

        return self.parse("""
raise $_.TypeError('Can\\'t convert %(type_name)s to pointer: %%r' %% $in_)
""" % {"type_name": type(self).__name__}, in_=name)["in_"]