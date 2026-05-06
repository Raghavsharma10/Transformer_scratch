def compute_max_min_depth(germanet_db):
    '''
    For every part of speech in GermaNet, computes the maximum
    min_depth in that hierarchy.

    Arguments:
    - `germanet_db`: a pymongo.database.Database object
    '''
    gnet           = germanet.GermaNet(germanet_db)
    max_min_depths = defaultdict(lambda: -1)
    for synset in gnet.all_synsets():
        min_depth = synset.min_depth
        if max_min_depths[synset.category] < min_depth:
            max_min_depths[synset.category] = min_depth

    if germanet_db.metainfo.count() == 0:
        germanet_db.metainfo.insert({})
    metainfo = germanet_db.metainfo.find_one()
    metainfo['max_min_depths'] = max_min_depths
    germanet_db.metainfo.save(metainfo)

    print('Computed maximum min_depth for all parts of speech:')
    print(u', '.join(u'{0}: {1}'.format(k, v) for (k, v) in
                     sorted(max_min_depths.items())).encode('utf-8'))