def before_insert(mapper, conn, target):
        """event.listen method for Sqlalchemy to set the seqience_id for this
        object and create an ObjectNumber value for the id_"""

        # from identity import ObjectNumber
        # assert not target.fk_vid or not ObjectNumber.parse(target.fk_vid).revision

        if target.sequence_id is None:
            from ambry.orm.exc import DatabaseError
            raise DatabaseError('Must have sequence_id before insertion')

        # Check that the id column is always sequence id 1
        assert (target.name == 'id') == (target.sequence_id == 1), (target.name, target.sequence_id)

        Column.before_update(mapper, conn, target)