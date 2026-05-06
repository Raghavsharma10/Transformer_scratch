def get_sqlalchemy_url(database=None, host=None, port=None, username=None, password=None, driver='postgres'):
        # type: (Optional[str], Optional[str], Union[int, str, None], Optional[str], Optional[str], str) -> str
        """Gets SQLAlchemy url from database connection parameters

        Args:
            database (Optional[str]): Database name
            host (Optional[str]): Host where database is located
            port (Union[int, str, None]): Database port
            username (Optional[str]): Username to log into database
            password (Optional[str]): Password to log into database
            driver (str): Database driver. Defaults to 'postgres'.

        Returns:
            db_url (str): SQLAlchemy url
        """
        strings = ['%s://' % driver]
        if username:
            strings.append(username)
            if password:
                strings.append(':%s@' % password)
            else:
                strings.append('@')
        if host:
            strings.append(host)
        if port is not None:
            strings.append(':%d' % int(port))
        if database:
            strings.append('/%s' % database)
        return ''.join(strings)