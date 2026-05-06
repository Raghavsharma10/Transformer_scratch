def readPrefs_dms_tools_format(f):
    """Reads the amino-acid preferences written by `dms_tools v1 <http://jbloomlab.github.io/dms_tools/>`_.

    This is an exact copy of the same code from
    `dms_tools.file_io.ReadPreferences`. It is copied because
    `dms_tools v1 <http://jbloomlab.github.io/dms_tools/>`_ is currently
    only compatible with `python2`, and we needed something that also works
    with `python3`.

    *f* is the name of an existing file or a readable file-like object.
    It should be in the format written by
    `dms_tools v1 <http://jbloomlab.github.io/dms_tools/>`_.

    The return value is the tuple: *(sites, wts, pi_means, pi_95credint, h)*
    where *sites*, *wts*, *pi_means*, and *pi_95credint* will all
    have the same values used to write the file with *WritePreferences*,
    and *h* is a dictionary with *h[r]* giving the site entropy (log base
    2) for each *r* in *sites*.
    """
    charmatch = re.compile('^PI_([A-z\*\-]+)$')
    if isinstance(f, str):
        f = open(f)
        lines = f.readlines()
        f.close()
    else:
        lines = f.readlines()
    characters = []
    sites = []
    wts = {}
    pi_means = {}
    pi_95credint = {}
    h = {}
    for line in lines:
        if line.isspace():
            continue
        elif line[0] == '#' and not characters:
            entries = line[1 : ].strip().split()
            if len(entries) < 4:
                raise ValueError("Insufficient entries in header:\n%s" % line)
            if not (entries[0] in ['POSITION', 'SITE'] and entries[1][ : 2] == 'WT' and entries[2] == 'SITE_ENTROPY'):
                raise ValueError("Not the correct first three header columns:\n%s" % line)
            i = 3
            while i < len(entries) and charmatch.search(entries[i]):
                characters.append(charmatch.search(entries[i]).group(1))
                i += 1
            if i  == len(entries):
                pi_95credint = None
                linelength = len(characters) + 3
            else:
                if not len(entries) - i == len(characters):
                    raise ValueError("Header line does not have valid credible interval format:\n%s" % line)
                if not all([entries[i + j] == 'PI_%s_95' % characters[j] for j in range(len(characters))]):
                    raise ValueError("mean and credible interval character mismatch in header:\n%s" % line)
                linelength = 2 * len(characters) + 3
        elif line[0] == '#':
            continue
        elif not characters:
            raise ValueError("Found data lines before encountering a valid header")
        else:
            entries = line.strip().split()
            if len(entries) != linelength:
                raise ValueError("Line does not have expected %d entries:\n%s" % (linelength, line))
            r = entries[0]
            assert r not in sites, "Duplicate site of %s" % r
            sites.append(r)
            wts[r] = entries[1]
            assert entries[1] in characters or entries[1] == '?', "Character %s is not one of the valid ones in header. Valid possibilities: %s" % (entries[1], ', '.join(characters))
            h[r] = float(entries[2])
            pi_means[r] = dict([(x, float(entries[3 + i])) for (i, x) in enumerate(characters)])
            if pi_95credint != None:
                pi_95credint[r] = dict([(x, (float(entries[3 + len(characters) + i].split(',')[0]), float(entries[3 + len(characters) + i].split(',')[1]))) for (i, x) in enumerate(characters)])
    return (sites, wts, pi_means, pi_95credint, h)