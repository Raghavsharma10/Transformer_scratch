def _fetchSequence(ac, startIndex=None, endIndex=None):
    """Fetch sequences from NCBI using the eself interface.

    An interbase interval may be optionally provided with startIndex and
    endIndex. NCBI eself will return just the requested subsequence, which
    might greatly reduce payload sizes (especially with chromosome-scale
    sequences). When wrapped is True, return list of sequence lines rather
    than concatenated sequence.

    >>> len(_fetchSequence('NP_056374.2'))
    1596

    Pass the desired interval rather than using Python's [] slice
    operator.

    >>> _fetchSequence('NP_056374.2',0,10)
    'MESRETLSSS'

    >>> _fetchSequence('NP_056374.2')[0:10]
    'MESRETLSSS'

    """
    urlFmt = (
        "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        "db=nucleotide&id={ac}&rettype=fasta&retmode=text")
    if startIndex is None or endIndex is None:
        url = urlFmt.format(ac=ac)
    else:
        urlFmt += "&seq_start={start}&seq_stop={stop}"
        url = urlFmt.format(ac=ac, start=startIndex + 1, stop=endIndex)
    resp = requests.get(url)
    resp.raise_for_status()
    seqlines = resp.content.splitlines()[1:]
    print("{ac}[{s},{e}) => {n} lines ({u})".format(
        ac=ac, s=startIndex, e=endIndex, n=len(seqlines), u=url))
    # return response as list of lines, already line wrapped
    return seqlines