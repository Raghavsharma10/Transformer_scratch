def set_mysql(host, user, password, db, charset):
    """Set the SQLAlchemy connection string with MySQL settings"""
    manager.database.set_mysql_connection(
        host=host,
        user=user,
        password=password,
        db=db,
        charset=charset
    )