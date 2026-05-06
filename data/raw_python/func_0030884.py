def _fetch_seq_ensembl(ac, start_i=None, end_i=None):
    """Fetch the specified sequence slice from Ensembl using the public
    REST interface.

    An interbase interval may be optionally provided with start_i and
    end_i. However, the Ensembl REST interface does not currently
    accept intervals, so the entire sequence is returned and sliced
    locally.

    >> len(_fetch_seq_ensembl('ENSP00000288602'))
    766

    >> _fetch_seq_ensembl('ENSP00000288602',0,10)
    u'MAALSGGGGG'

    >> _fetch_seq_ensembl('ENSP00000288602')[0:10]
    u'MAALSGGGGG'

    >> ac = 'ENSP00000288602'
    >> _fetch_seq_ensembl(ac ,0, 10) == _fetch_seq_ensembl(ac)[0:10]
    True

    """

    url_fmt = "http://rest.ensembl.org/sequence/id/{ac}"
    url = url_fmt.format(ac=ac)
    r = requests.get(url, headers={"Content-Type": "application/json"})
    r.raise_for_status()
    seq = r.json()["seq"]
    return seq if (start_i is None or end_i is None) else seq[start_i:end_i]