def get_longest_target_alignment_coords_by_name(self,name):
    """For a name get the best alignment

    :return: [filebyte,innerbyte] describing the to distance the zipped block start, and the distance within the unzipped block
    :rtype: list
    """
    longest = -1
    coord = None
    #for x in self._queries[self._name_to_num[name]]:
    for line in [self._lines[x] for x in self._name_to_num[name]]:
      if line['flag'] & 2304 == 0: 
        return [line['filestart'],line['innerstart']]
    return None
    sys.stderr.write("ERROR: no primary alignment set in index\n")
    sys.exit()