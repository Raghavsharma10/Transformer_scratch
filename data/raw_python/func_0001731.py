def get_params_from_sqlalchemy_url(db_url):
        # type: (str) -> Dict[str,Any]
        """Gets PostgreSQL database connection parameters from SQLAlchemy url

        Args:
            db_url (str): SQLAlchemy url

        Returns:
            Dict[str,Any]: Dictionary of database connection parameters
        """
        result = urlsplit(db_url)
        return {'database': result.path[1:], 'host': result.hostname, 'port': result.port,
                'username': result.username, 'password': result.password, 'driver': result.scheme}