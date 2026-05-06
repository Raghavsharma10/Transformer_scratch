def writeAnn(filename, catalog, fmt):
    """
    Write an annotation file that can be read by Kvis (.ann) or DS9 (.reg).
    Uses ra/dec from catalog.
    Draws ellipses if bmaj/bmin/pa are in catalog. Draws 30" circles otherwise.

    Only :class:`AegeanTools.models.OutputSource` will appear in the annotation file
    unless there are none, in which case :class:`AegeanTools.models.SimpleSource` (if present)
    will be written. If any :class:`AegeanTools.models.IslandSource` objects are present then
    an island contours file will be written.

    Parameters
    ----------
    filename : str
        Output filename base.

    catalog : list
        List of sources.

    fmt : ['ann', 'reg']
        Output file type.

    Returns
    -------
    None

    See Also
    --------
    AegeanTools.catalogs.writeIslandContours
    """
    if fmt not in ['reg', 'ann']:
        log.warning("Format not supported for island boxes{0}".format(fmt))
        return  # fmt not supported

    components, islands, simples = classify_catalog(catalog)
    if len(components) > 0:
        cat = sorted(components)
        suffix = "comp"
    elif len(simples) > 0:
        cat = simples
        suffix = "simp"
    else:
        cat = []

    if len(cat) > 0:
        ras = [a.ra for a in cat]
        decs = [a.dec for a in cat]
        if not hasattr(cat[0], 'a'):  # a being the variable that I used for bmaj.
            bmajs = [30 / 3600.0 for a in cat]
            bmins = bmajs
            pas = [0 for a in cat]
        else:
            bmajs = [a.a / 3600.0 for a in cat]
            bmins = [a.b / 3600.0 for a in cat]
            pas = [a.pa for a in cat]

        names = [a.__repr__() for a in cat]
        if fmt == 'ann':
            new_file = re.sub('.ann$', '_{0}.ann'.format(suffix), filename)
            out = open(new_file, 'w')
            print("#Aegean version {0}-({1})".format(__version__, __date__), file=out)
            print('PA SKY', file=out)
            print('FONT hershey12', file=out)
            print('COORD W', file=out)
            formatter = "ELLIPSE W {0} {1} {2} {3} {4:+07.3f} #{5}\nTEXT W {0} {1} {5}"
        else:  # reg
            new_file = re.sub('.reg$', '_{0}.reg'.format(suffix), filename)
            out = open(new_file, 'w')
            print("#Aegean version {0}-({1})".format(__version__, __date__), file=out)
            print("fk5", file=out)
            formatter = 'ellipse {0} {1} {2:.9f}d {3:.9f}d {4:+07.3f}d # text="{5}"'
            # DS9 has some strange ideas about position angle
            pas = [a - 90 for a in pas]

        for ra, dec, bmaj, bmin, pa, name in zip(ras, decs, bmajs, bmins, pas, names):
            # comment out lines that have invalid or stupid entries
            if np.nan in [ra, dec, bmaj, bmin, pa] or bmaj >= 180:
                print('#', end=' ', file=out)
            print(formatter.format(ra, dec, bmaj, bmin, pa, name), file=out)
        out.close()
        log.info("wrote {0}".format(new_file))
    if len(islands) > 0:
        if fmt == 'reg':
            new_file = re.sub('.reg$', '_isle.reg', filename)
        elif fmt == 'ann':
            log.warning('kvis islands are currently not working')
            return
        else:
            log.warning('format {0} not supported for island annotations'.format(fmt))
            return
        writeIslandContours(new_file, islands, fmt)
        log.info("wrote {0}".format(new_file))

    return