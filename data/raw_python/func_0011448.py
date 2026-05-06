def init_db(self):
    '''initialize the database, with the default database path or custom with
       a format corresponding to the database type:

       Examples:

       sqlite:////scif/data/expfactory.db
    '''

    # The user can provide a custom string
    if self.database is None:
        self.logger.error("You must provide a database url, exiting.")
        sys.exit(1)

    self.engine = create_engine(self.database, convert_unicode=True)
    self.session = scoped_session(sessionmaker(autocommit=False,
                                               autoflush=False,
                                               bind=self.engine))

    # Database Setup
    Base.query = self.session.query_property()

    # import all modules here that might define models so that
    # they will be registered properly on the metadata.  Otherwise
    # you will have to import them first before calling init_db()
    import expfactory.database.models
    self.Base = Base
    self.Base.metadata.create_all(bind=self.engine)