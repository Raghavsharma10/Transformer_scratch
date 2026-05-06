def is_indexed(self, identifier):
        """ Returns True if identifier is already indexed. Otherwise returns False. """
        query = text("""
            SELECT identifier
            FROM identifier_index
            WHERE identifier = :identifier;
        """)
        result = self.backend.library.database.connection.execute(query, identifier=identifier['identifier'])
        return bool(result.fetchall())