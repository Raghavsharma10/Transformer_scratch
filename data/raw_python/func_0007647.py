def get(self):
        """
        Run the query and return a `Report`.

        This method transparently handles paginated results, so even for results that
        are larger than the maximum amount of rows the Google Analytics API will
        return in a single request, or larger than the amount of rows as specified
        through `CoreQuery#step`, `get` will leaf through all pages,
        concatenate the results and produce a single Report instance.
        """

        cursor = self
        report = None
        is_complete = False
        is_enough = False

        while not (is_enough or is_complete):
            chunk = cursor.execute()

            if report:
                report.append(chunk.raw[0], cursor)
            else:
                report = chunk

            is_enough = len(report.rows) >= self.meta.get('limit', float('inf'))
            is_complete = chunk.is_complete
            cursor = cursor.next()

        return report