def read_entry(self):
    """get the next value from the array, and set internal iterator so next call will be next entry

    :return: The next GenomicRange entry
    :rtype: GenomicRange
    """
    if len(self.bedarray) <= self.curr_ind: return None
    val = self.bedarray[self.curr_ind]
    self.curr_ind += 1
    return val