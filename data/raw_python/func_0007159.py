def initialize_connections(self, scopefunc=None):
        """
        Initialize a database connection by each connection string
        defined in the configuration file
        """
        for connection_name, connection_string in\
                self.app.config['FLASK_PHILO_SQLALCHEMY'].items():
            engine = create_engine(connection_string)
            session = scoped_session(sessionmaker(), scopefunc=scopefunc)
            session.configure(bind=engine)
            self.connections[connection_name] = Connection(engine, session)