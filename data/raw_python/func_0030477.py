def before_insert(mapper, conn, target):
        """event.listen method for Sqlalchemy to set the sequence for this
        object and create an ObjectNumber value for the id_"""
        from sqlalchemy import text

        if not target.id:
            sql = text('SELECT max(f_id)+1 FROM files WHERE f_d_vid = :did')

            target.id, = conn.execute(sql, did=target.d_vid).fetchone()

            if not target.id:
                target.id = 1

        if target.contents and isinstance(target.contents, six.text_type):
            target.contents = target.contents.encode('utf-8')

        File.before_update(mapper, conn, target)