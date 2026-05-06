def before_insert(mapper, conn, target):
        """event.listen method for Sqlalchemy to set the seqience_id for this
        object and create an ObjectNumber value for the id"""
        if target.sequence_id is None:
            from ambry.orm.exc import DatabaseError
            raise DatabaseError('Must have sequence id before insertion')

        Table.before_update(mapper, conn, target)