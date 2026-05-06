def initialize(address='127.0.0.1:27017', database_name='hfos', instance_name="default", reload=False):
    """Initializes the database connectivity, schemata and finally object models"""

    global schemastore
    global l10n_schemastore
    global objectmodels
    global collections
    global dbhost
    global dbport
    global dbname
    global instance
    global initialized

    if initialized and not reload:
        hfoslog('Already initialized and not reloading.', lvl=warn, emitter="DB", frame_ref=2)
        return

    dbhost = address.split(':')[0]
    dbport = int(address.split(":")[1]) if ":" in address else 27017
    dbname = database_name

    db_log("Using database:", dbname, '@', dbhost, ':', dbport)

    try:
        client = pymongo.MongoClient(host=dbhost, port=dbport)
        db = client[dbname]
        db_log("Database: ", db.command('buildinfo'), lvl=debug)
    except Exception as e:
        db_log("No database available! Check if you have mongodb > 3.0 "
               "installed and running as well as listening on port 27017 "
               "of localhost. (Error: %s) -> EXIT" % e, lvl=critical)
        sys.exit(5)

    warmongo.connect(database_name)

    schemastore = _build_schemastore_new()
    l10n_schemastore = _build_l10n_schemastore(schemastore)
    objectmodels = _build_model_factories(schemastore)
    collections = _build_collections(schemastore)
    instance = instance_name
    initialized = True