def before_insert(mapper, conn, target):
        """event.listen method for Sqlalchemy to set the sequence for this
        object and create an ObjectNumber value for the id_"""

        target._set_ids()

        if target.name and target.vname and target.cache_key and target.fqname and not target.dataset:
            return

        Partition.before_update(mapper, conn, target)