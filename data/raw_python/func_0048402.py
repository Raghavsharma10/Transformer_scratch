def get_unaligned_lines(self):
    """get the lines that are not aligned"""
    sys.stderr.write("error unimplemented get_unaligned_lines\n")
    sys.exit()
    return [self._lines[x-1] for x in self._unaligned]