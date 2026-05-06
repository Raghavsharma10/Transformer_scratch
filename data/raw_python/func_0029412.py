def migrate_codec(config_old, config_new):
    '''Migrate data from mongodict <= 0.2.1 to 0.3.0
    `config_old` and `config_new` should be dictionaries with the keys
    regarding to MongoDB server:
        - `host`
        - `port`
        - `database`
        - `collection`
    '''
    assert mongodict.__version__ in [(0, 3, 0), (0, 3, 1)]

    connection = pymongo.Connection(host=config_old['host'],
                                    port=config_old['port'])
    database = connection[config_old['database']]
    collection = database[config_old['collection']]
    new_dict = mongodict.MongoDict(**config_new) # uses pickle codec by default
    total_pairs = collection.count()
    start_time = time.time()
    for counter, pair in enumerate(collection.find(), start=1):
        key, value = pair['_id'], pair['value']
        new_dict[key] = value
        if counter % REPORT_INTERVAL == 0:
            print_report(counter, total_pairs, start_time)
    print_report(counter, total_pairs, start_time)
    print('')