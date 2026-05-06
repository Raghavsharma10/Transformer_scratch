def insert_infocontent_data(germanet_db):
    '''
    For every synset in GermaNet, inserts count information derived
    from SDEWAC.

    Arguments:
    - `germanet_db`: a pymongo.database.Database object
    '''
    gnet           = germanet.GermaNet(germanet_db)
    # use add one smoothing
    gn_counts      = defaultdict(lambda: 1.)
    total_count    = 1
    input_file     = gzip.open(os.path.join(os.path.dirname(__file__),
                                            WORD_COUNT_FILE))
    num_lines_read = 0
    num_lines      = 0
    for line in input_file:
        line       = line.decode('utf-8').strip().split('\t')
        num_lines += 1
        if len(line) != 3:
            continue
        count, pos, word = line
        num_lines_read += 1
        count           = int(count)
        synsets         = set(gnet.synsets(word, pos))
        if not synsets:
            continue
        # Although Resnik (1995) suggests dividing count by the number
        # of synsets, Patwardhan et al (2003) argue against doing
        # this.
        count = float(count) / len(synsets)
        for synset in synsets:
            total_count += count
            paths = synset.hypernym_paths
            scount = float(count) / len(paths)
            for path in paths:
                for ss in path:
                    gn_counts[ss._id] += scount
    print('Read {0} of {1} lines from count file.'.format(num_lines_read,
                                                          num_lines))
    print('Recorded counts for {0} synsets.'.format(len(gn_counts)))
    print('Total count is {0}'.format(total_count))
    input_file.close()
    # update all the synset records in GermaNet
    num_updates = 0
    for synset in germanet_db.synsets.find():
        synset['infocont'] = gn_counts[synset['_id']] / total_count
        germanet_db.synsets.save(synset)
        num_updates += 1
    print('Updated {0} synsets.'.format(num_updates))