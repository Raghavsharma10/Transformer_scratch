def next_sequence_id(self, parent_table_class, parent_vid, child_table_class):
        """Get the next sequence id for child objects for a parent object that has a child sequence
        field"""

        from sqlalchemy import text

        # Name of sequence id column in the child
        c_seq_col = child_table_class.sequence_id.property.columns[0].name

        p_seq_col = getattr(parent_table_class, c_seq_col).property.columns[0].name

        p_vid_col = parent_table_class.vid.property.columns[0].name

        if self.driver == 'sqlite':
            # The Sqlite version is not atomic, but Sqlite also doesn't support concurrency
            # So, we don't have to open a new connection, but we also can't open a new connection, so
            # this uses the session.
            self.commit()
            sql = text("SELECT  {p_seq_col} FROM {p_table} WHERE {p_vid_col} = '{parent_vid}' "
                       .format(p_table=parent_table_class.__tablename__, p_seq_col=p_seq_col,
                               p_vid_col=p_vid_col, parent_vid=parent_vid))

            v = next(iter(self.session.execute(sql)))[0]
            sql = text("UPDATE {p_table} SET {p_seq_col} = {p_seq_col} + 1 WHERE {p_vid_col} = '{parent_vid}' "
                       .format(p_table=parent_table_class.__tablename__, p_seq_col=p_seq_col,
                               p_vid_col=p_vid_col, parent_vid=parent_vid))

            self.session.execute(sql)
            self.commit()
            return v

        else:
            # Must be postgres, or something else that supports "RETURNING"
            sql = text("""
            UPDATE {p_table} SET {p_seq_col} = {p_seq_col} + 1 WHERE {p_vid_col} = '{parent_vid}' RETURNING {p_seq_col}
            """.format(p_table=parent_table_class.__tablename__, p_seq_col=p_seq_col, p_vid_col=p_vid_col,
                       parent_vid=parent_vid))

            self.connection.execute('SET search_path TO {}'.format(self._schema))
            r = self.connection.execute(sql)
            v = next(iter(r))[0]
            return v-1