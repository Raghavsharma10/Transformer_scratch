def query_sequence(self):
    """ Overrides align. corrects orientation with reverse complement if on negative strand

    .. warning:: this returns the full query sequence, not just the aligned portion, but i also does not include hard clipped portions (only soft clipped)
    """
    if not self.entries.seq: return None
    if self.check_flag(0x10): return rc(self.entries.seq)
    return self.entries.seq