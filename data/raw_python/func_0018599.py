def get_opcodes(matching_blocks):
    """Use difflib to get the opcodes for a set of matching blocks."""
    sm = difflib.SequenceMatcher(a=[], b=[])
    sm.matching_blocks = matching_blocks
    return sm.get_opcodes()