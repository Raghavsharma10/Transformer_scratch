def clear_all():
    """DANGER!
    *This command is a maintenance tool and clears the complete database.*
    """

    sure = input("Are you sure to drop the complete database content? (Type "
                 "in upppercase YES)")
    if not (sure == 'YES'):
        db_log('Not deleting the database.')
        sys.exit(5)

    client = pymongo.MongoClient(host=dbhost, port=dbport)
    db = client[dbname]

    for col in db.collection_names(include_system_collections=False):
        db_log("Dropping collection ", col, lvl=warn)
        db.drop_collection(col)