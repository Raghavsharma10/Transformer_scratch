def get_coords_by_name(self,name):
    """
    .. warning:: not implemented
    """
    sys.stderr.write("error unimplemented get_coords_by_name\n")
    sys.exit()
    return [[self._lines[x]['filestart'],self._lines[x]['innerstart']] for x in self._queries[self._name_to_num[name]]]