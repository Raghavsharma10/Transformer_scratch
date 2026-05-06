def get_session(db_url):
        # type: (str) -> Session
        """Gets SQLAlchemy session given url. Your tables must inherit
        from Base in hdx.utilities.database.

        Args:
            db_url (str): SQLAlchemy url

        Returns:
            sqlalchemy.orm.session.Session: SQLAlchemy session
        """
        engine = create_engine(db_url, poolclass=NullPool, echo=False)
        Session = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        return Session()