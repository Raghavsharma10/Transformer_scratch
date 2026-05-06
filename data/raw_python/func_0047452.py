def get_range_string(self):
    """ get the range string represetation. similar to the default input for UCSC genome browser

    :return: representation by string like chr2:801-900
    :rtype: string
    """
    return self.chr+":"+str(self.start)+"-"+str(self.end)