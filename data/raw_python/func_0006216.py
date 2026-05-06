def setup_db(connection_string):
    """
        Sets up the database schema and adds defaults.
    :param connection_string: Database URL. e.g: sqlite:///filename.db
                              This is usually taken from the config file.
    """
    global DB_Session, engine
    new_database = False
    if connection_string == 'sqlite://' or not database_exists(connection_string):
        new_database = True
    engine = create_engine(connection_string, connect_args={'timeout': 20})
    entities.Base.metadata.create_all(engine)
    DB_Session = sessionmaker(bind=engine)
    db_path = os.path.dirname(__file__)

    if new_database:
        # bootstrapping the db with classifications types.
        json_file = open(os.path.join(db_path, 'bootstrap.json'))
        data = json.load(json_file)
        session = get_session()
        session.execute('PRAGMA user_version = {0}'.format(beeswarm.server.db.DATABASE_VERSION))
        for entry in data['classifications']:
            c = session.query(Classification).filter(Classification.type == entry['type']).first()
            if not c:
                classification = Classification(type=entry['type'], description_short=entry['description_short'],
                                                description_long=entry['description_long'])
                session.add(classification)
            else:
                c.description_short = entry['description_short']
                c.description_long = entry['description_long']
        for username in data['bait_users']:
            u = session.query(BaitUser).filter(BaitUser.username == username).first()
            if not u:
                logger.debug('Creating default BaitUser: {}'.format(username))
                password = data['bait_users'][username]
                bait_user = BaitUser(username=username, password=password)
                session.add(bait_user)
        session.commit()
    else:
        result = engine.execute("PRAGMA user_version;")
        version = result.fetchone()[0]
        result.close()
        logger.info('Database is at version {0}.'.format(version))
        if version != beeswarm.server.db.DATABASE_VERSION:
            logger.error('Incompatible database version detected. This version of Beeswarm is compatible with '
                         'database version {0}, but {1} was found. Please delete the database, restart the Beeswarm '
                         'server and reconnect the drones.'.format(beeswarm.server.db.DATABASE_VERSION,
                                                                   version))
            sys.exit(1)