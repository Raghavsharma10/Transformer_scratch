def equals(self,gr):
    """ check for equality. does not consider direction

    :param gr: another genomic range
    :type gr: GenomicRange
    :return: true if they are the same, false if they are not
    :rtype: bool
    """
    if self.chr == gr.chr and self.start == gr.start and self.end == gr.end:
      return True
    return False