def init_db(self):
    '''initialize the database, with the default database path or custom of
       the format sqlite:////scif/data/expfactory.db

    '''

    # Database Setup, use default if uri not provided
    if self.database == 'sqlite':
        db_path = os.path.join(EXPFACTORY_DATA, '%s.db' % EXPFACTORY_SUBID)
        self.database = 'sqlite:///%s' % db_path

    bot.info("Database located at %s" % self.database)
    self.engine = create_engine(self.database, convert_unicode=True)
    self.session = scoped_session(sessionmaker(autocommit=False,
                                               autoflush=False,
                                               bind=self.engine))
    
    Base.query = self.session.query_property()

    # import all modules here that might define models so that
    # they will be registered properly on the metadata.  Otherwise
    # you will have to import them first before calling init_db()
    import expfactory.database.models
    Base.metadata.create_all(bind=self.engine)
    self.Base = Base