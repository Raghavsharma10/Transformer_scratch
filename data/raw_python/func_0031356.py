def _debugPrint(hsp, queryLen, localDict, msg=''):
    """
    Print debugging information showing the local variables used during
    a call to normalizeHSP and the hsp and then raise an C{AssertionError}.

    @param hsp: The HSP C{dict} passed to normalizeHSP.
    @param queryLen: the length of the query sequence.
    @param localDict: A C{dict} of local variables (as produced by locals()).
    @param msg: A C{str} message to raise C{AssertionError} with.
    @raise AssertionError: unconditionally.
    """
    print('normalizeHSP error:', file=sys.stderr)
    print('  queryLen: %d' % queryLen, file=sys.stderr)

    print('  Original HSP:', file=sys.stderr)
    for attr in ['bits', 'btop', 'expect', 'frame', 'query_end', 'query_start',
                 'sbjct', 'query', 'sbjct_end', 'sbjct_start']:
        print('    %s: %r' % (attr, hsp[attr]), file=sys.stderr)

    print('  Local variables:', file=sys.stderr)
    for var in sorted(localDict):
        if var != 'hsp':
            print('    %s: %s' % (var, localDict[var]), file=sys.stderr)

    raise AssertionError(msg)