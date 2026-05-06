def update(connection, force_download):
    """Update the database"""
    manager.database.update(
        connection=connection,
        force_download=force_download
    )