def query(self, term):
        """ Run a query.

        :arg str term: The query term to run.

        :Returns: A list of results as :class:`pygerrit.models.Change` objects.

        :Raises: `ValueError` if `term` is not a string.

        """
        results = []
        command = ["query", "--current-patch-set", "--all-approvals",
                   "--format JSON", "--commit-message"]

        if not isinstance(term, basestring):
            raise ValueError("term must be a string")

        command.append(escape_string(term))
        result = self._ssh_client.run_gerrit_command(" ".join(command))
        decoder = JSONDecoder()
        for line in result.stdout.read().splitlines():
            # Gerrit's response to the query command contains one or more
            # lines of JSON-encoded strings.  The last one is a status
            # dictionary containing the key "type" whose value indicates
            # whether or not the operation was successful.
            # According to http://goo.gl/h13HD it should be safe to use the
            # presence of the "type" key to determine whether the dictionary
            # represents a change or if it's the query status indicator.
            try:
                data = decoder.decode(line)
            except ValueError as err:
                raise GerritError("Query returned invalid data: %s", err)
            if "type" in data and data["type"] == "error":
                raise GerritError("Query error: %s" % data["message"])
            elif "project" in data:
                results.append(Change(data))
        return results