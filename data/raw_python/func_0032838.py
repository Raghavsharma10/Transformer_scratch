def table_names(self):
        """Returns names of all tables in the database"""
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        cursor = self.connection.execute(query)
        results = cursor.fetchall()
        return [result_tuple[0] for result_tuple in results]