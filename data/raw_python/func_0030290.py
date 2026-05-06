def table_convert_geometry(metadata, table_name):
    """Get table metadata from the database."""
    from sqlalchemy import Table
    from ..orm import Geometry

    table = Table(table_name, metadata, autoload=True)

    for c in table.columns:

        # HACK! Sqlalchemy sees spatialte GEOMETRY types
        # as NUMERIC

        if c.name == 'geometry':
            c.type = Geometry # What about variants?

    return table