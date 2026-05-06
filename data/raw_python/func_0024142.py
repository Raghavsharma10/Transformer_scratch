def find_issues(self, criteria={}, jql=None, order='KEY ASC', verbose=False, changelog=True):
        """Return a list of issues with changelog metadata.

        Searches for the `issue_types`, `project`, `valid_resolutions` and
        'jql_filter' set in the passed-in `criteria` object.

        Pass a JQL string to further qualify the query results.
        """

        query = []

        if criteria.get('project', False):
            query.append('project IN (%s)' % ', '.join(['"%s"' % p for p in criteria['project']]))

        if criteria.get('issue_types', False):
            query.append('issueType IN (%s)' % ', '.join(['"%s"' % t for t in criteria['issue_types']]))

        if criteria.get('valid_resolutions', False):
            query.append('(resolution IS EMPTY OR resolution IN (%s))' % ', '.join(['"%s"' % r for r in criteria['valid_resolutions']]))

        if criteria.get('jql_filter') is not None:
            query.append('(%s)' % criteria['jql_filter'])

        if jql is not None:
            query.append('(%s)' % jql)

        queryString = "%s ORDER BY %s" % (' AND '.join(query), order,)

        if verbose:
            print("Fetching issues with query:", queryString)

        fromRow=0
        issues = []
        while True:
            try:
                if changelog:
                    pageofissues = self.jira.search_issues(queryString, expand='changelog', maxResults=self.settings['max_results'],startAt=fromRow)
                else:
                    pageofissues = self.jira.search_issues(queryString, maxResults=self.settings['max_results'],startAt=fromRow)

                fromRow = fromRow + int(self.settings['max_results'])
                issues += pageofissues
                if verbose:
                    print("Got %s lines per jira query from result starting at line number %s " % (self.settings['max_results'],  fromRow))
                if len(pageofissues)==0:
                    break
            except JIRAError as e:
                print("Jira query error with: {}\n{}".format(queryString, e))
                return []


        if verbose:
            print("Fetched", len(issues), "issues")

        return issues