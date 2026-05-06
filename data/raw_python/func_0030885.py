def _fetch_seq_ncbi(ac, start_i=None, end_i=None):
    """Fetch sequences from NCBI using the eutils interface.

    An interbase interval may be optionally provided with start_i and
    end_i. NCBI eutils will return just the requested subsequence,
    which might greatly reduce payload sizes (especially with
    chromosome-scale sequences).

    The request includes `tool` and `email` arguments to identify the
    caller as the bioutils package.  According to
    https://www.ncbi.nlm.nih.gov/books/NBK25497/, these values should
    correspond to the library, not the library client.  Using the
    defaults is recommended.  Nonetheless, callers may set
    `bioutils.seqfetcher.ncbi_tool` and
    `bioutils.seqfetcher.ncbi_email` to custom values if that is
    desired.


    >> len(_fetch_seq_ncbi('NP_056374.2'))
    1596

    Pass the desired interval rather than using Python's [] slice
    operator.

    >> _fetch_seq_ncbi('NP_056374.2',0,10)
    'MESRETLSSS'

    >> _fetch_seq_ncbi('NP_056374.2')[0:10]
    'MESRETLSSS'

    >> _fetch_seq_ncbi('NP_056374.2',0,10) == _fetch_seq_ncbi('NP_056374.2')[0:10]
    True

    """

    db = "protein" if ac[1] == "P" else "nucleotide"
    url_fmt = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
               "db={db}&id={ac}&rettype=fasta")

    if start_i is None or end_i is None:
        url = url_fmt.format(db=db, ac=ac)
    else:
        url_fmt += "&seq_start={start}&seq_stop={stop}"
        url = url_fmt.format(db=db, ac=ac, start=start_i + 1, stop=end_i)

    url += "&tool={tool}&email={email}".format(tool=ncbi_tool, email=ncbi_email)

    url = _add_eutils_api_key(url)

    n_retries = 0
    while True:
        resp = requests.get(url)
        if resp.ok:
            seq = "".join(resp.text.splitlines()[1:])
            return seq
        if n_retries >= retry_limit:
            break
        if n_retries == 0:
            _logger.warning("Failed to fetch {}".format(url))
        sleeptime = random.randint(n_retries,3) ** n_retries
        _logger.warning("Failure {}/{}; retry in {} seconds".format(n_retries, retry_limit, sleeptime))
        time.sleep(sleeptime)
        n_retries += 1
    # Falls through only on failure
    resp.raise_for_status()