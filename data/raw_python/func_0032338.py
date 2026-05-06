def dump(self, as_json=True, commit_batch_size=100):
        """
        Dump the grant information to a local storage.

        :param as_json: Convert XML to JSON before saving (default: True).
        """
        connection = sqlite3.connect(self.destination)
        format_ = 'json' if as_json else 'xml'
        if not self._db_exists(connection):
            connection.execute(
                "CREATE TABLE grants (data text, format text)")

        # This will call the RemoteOAIRELoader.iter_grants and fetch
        # records from remote location.
        grants_iterator = self.loader.iter_grants(as_json=as_json)
        for idx, grant_data in enumerate(grants_iterator, 1):
            if as_json:
                grant_data = json.dumps(grant_data, indent=2)
            connection.execute(
                "INSERT INTO grants VALUES (?, ?)", (grant_data, format_))

            # Commit to database every N records
            if idx % commit_batch_size == 0:
                connection.commit()
        connection.commit()
        connection.close()