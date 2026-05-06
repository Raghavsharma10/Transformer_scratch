def dbsource(dbname, var, resolution=None, tscale=None):
    """Return which file(s) to use according to dbname, var, etc
    """
    db_cfg = {}
    cfg_dir = 'datasource'
    cfg_files = pkg_resources.resource_listdir('oceansdb', cfg_dir)
    cfg_files = [f for f in cfg_files if f[-5:] == '.json']
    for src_cfg in cfg_files:
        text = pkg_resources.resource_string(
                'oceansdb', os.path.join(cfg_dir, src_cfg))
        text = text.decode('UTF-8', 'replace')
        cfg = json.loads(text)
        for c in cfg:
            assert c not in db_cfg, "Trying to overwrite %s"
            db_cfg[c] = cfg[c]

    dbpath = oceansdb_dir()
    datafiles = []
    cfg = db_cfg[dbname]

    if (resolution is None):
        resolution = cfg['vars'][var]['default_resolution']

    if (tscale is None):
        tscale = cfg['vars'][var][resolution]["default_tscale"]

    for c in cfg['vars'][var][resolution][tscale]:
        download_file(outputdir=dbpath, **c)

        if 'filename' in c:
            filename = os.path.join(dbpath, c['filename'])
        else:
            filename = os.path.join(dbpath,
                    os.path.basename(urlparse(c['url']).path))

        if 'varnames' in cfg['vars'][var][resolution]:
            datafiles.append(Dataset_flex(filename,
                aliases=cfg['vars'][var][resolution]['varnames']))
        else:
            datafiles.append(Dataset_flex(filename))

    return datafiles