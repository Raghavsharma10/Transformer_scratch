def query_quality(self):
    """ Overrides align

    .. warning:: this returns the full query quality, not just the aligned portion
    """
    if not self.entries.qual: return None
    if self.entries.qual == '*': return None
    if self.check_flag(0x10): return self.entries.qual[::-1]
    return self.entries.qual