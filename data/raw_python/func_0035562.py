def readPrefs(prefsfile, minpref=0, avgprefs=False, randprefs=False,
        seed=1, sites_as_strings=False):
    """Read preferences from file with some error checking.

    Args:
        `prefsfile` (string or readable file-like object)
            File holding amino-acid preferences. Can be
            comma-, space-, or tab-separated file with column
            headers of `site` and then all one-letter amino-acid
            codes, or can be in the more complex format written
            `dms_tools v1 <http://jbloomlab.github.io/dms_tools/>`_.
            Must be prefs for consecutively numbered sites starting at 1.
            Stop codon prefs can be present (stop codons are indicated by
            ``*``); if so they are removed and prefs re-normalized to sum to 1.
        `minpref` (float >= 0)
            Adjust all preferences to be >= this number.
        `avgprefs`, `randprefs` (bool)
            Mutually exclusive options specifying to average or
            randomize prefs across sites.
        `seed` (int)
            Seed used to sort random number generator for `randprefs`.
        `sites_as_strings` (bool)
            By default, the site numers are coerced to integers.
            If this option is `True`, then they are kept as strings.

    Returns:
        `prefs` (dict)
            `prefs[r][a]` is the preference of site `r` for amino-acid `a`.
            `r` is an `int` unless `sites_as_strings=True`.
    """
    assert minpref >= 0, 'minpref must be >= 0'

    aas = set(phydmslib.constants.AA_TO_INDEX.keys())

    try:
        df = pandas.read_csv(prefsfile, sep=None, engine='python')
        pandasformat = True
    except ValueError:
        pandasformat = False
    if pandasformat and (set(df.columns) == aas.union(set(['site'])) or
            set(df.columns) == aas.union(set(['site', '*']))):
        # read valid preferences as data frame
        sites = df['site'].tolist()
        prefs = {}
        for r in sites:
            rdf = df[df['site'] == r]
            prefs[r] = {}
            for aa in df.columns:
                if aa != 'site':
                    prefs[r][aa] = float(rdf[aa])
    else:
        # try reading as dms_tools format
        prefs = phydmslib.file_io.readPrefs_dms_tools_format(prefsfile)[2]
        sites = list(prefs.keys())

    # error check prefs
    if not sites_as_strings:
        try:
            sites = [int(r) for r in sites]
        except ValueError:
            raise ValueError("sites not int in prefsfile {0}".format(prefsfile))
        assert (min(sites) == 1 and max(sites) - min(sites) == len(sites) - 1),\
                "Sites not consecutive starting at 1"
        prefs = dict([(int(r), rprefs) for (r, rprefs) in prefs.items()])
    else:
        sites = [str(r) for r in sites]
        prefs = dict([(str(r), rprefs) for (r, rprefs) in prefs.items()])

    assert len(set(sites)) == len(sites), "Non-unique sites in prefsfiles"
    assert all([all([pi >= 0 for pi in rprefs.values()]) for rprefs in
            prefs.values()]), "prefs < 0 in prefsfile {0}".format(prefsfile)
    for r in list(prefs.keys()):
        rprefs = prefs[r]
        assert sum(rprefs.values()) - 1 <= 0.01, (
            "Prefs in prefsfile {0} don't sum to one".format(prefsfile))
        if '*' in rprefs:
            del rprefs['*']
        assert aas == set(rprefs.keys()), ("prefsfile {0} does not include "
                "all amino acids at site {1}").format(prefsfile, r)
        rsum = float(sum(rprefs.values()))
        prefs[r] = dict([(aa, pi / rsum) for (aa, pi) in rprefs.items()])
    assert set(sites) == set(prefs.keys())

    # Iteratively adjust until all prefs exceed minpref after re-scaling.
    for r in list(prefs.keys()):
        rprefs = prefs[r]
        iterations = 0
        while any([pi < minpref for pi in rprefs.values()]):
            rprefs = dict([(aa, max(1.1 * minpref,
                    pi)) for (aa, pi) in rprefs.items()])
            newsum = float(sum(rprefs.values()))
            rprefs = dict([(aa, pi / newsum) for (aa, pi) in rprefs.items()])
            iterations += 1
            assert iterations <= 3, "minpref adjustment not converging."
        prefs[r] = rprefs

    if randprefs:
        assert not avgprefs, "randprefs and avgprefs are incompatible"
        random.seed(seed)
        sites = sorted([r for r in prefs.keys()])
        prefs = [prefs[r] for r in sites]
        random.shuffle(sites)
        prefs = dict(zip(sites, prefs))
    elif avgprefs:
        avg_prefs = dict([(aa, 0.0) for aa in aas])
        for rprefs in prefs.values():
            for aa in aas:
                avg_prefs[aa] += rprefs[aa]
        for aa in aas:
            avg_prefs[aa] /= float(len(prefs))
        for r in list(prefs.keys()):
            prefs[r] = avg_prefs

    return prefs