def retrieve_info(self):
        """Query Bugzilla API to retrieve the needed infos."""

        scheme = urlparse(self.url).scheme
        netloc = urlparse(self.url).netloc
        query = urlparse(self.url).query

        if scheme not in ('http', 'https'):
            return

        for item in query.split('&'):
            if 'id=' in item:
                ticket_id = item.split('=')[1]
                break
        else:
            return

        bugzilla_url = '%s://%s/%s%s' % (scheme, netloc, _URI_BASE, ticket_id)

        result = requests.get(bugzilla_url)
        self.status_code = result.status_code

        if result.status_code == 200:
            tree = ElementTree.fromstring(result.content)

            self.title = tree.findall("./bug/short_desc").pop().text
            self.issue_id = tree.findall("./bug/bug_id").pop().text
            self.reporter = tree.findall("./bug/reporter").pop().text
            self.assignee = tree.findall("./bug/assigned_to").pop().text
            self.status = tree.findall("./bug/bug_status").pop().text
            self.product = tree.findall("./bug/product").pop().text
            self.component = tree.findall("./bug/component").pop().text
            self.created_at = tree.findall("./bug/creation_ts").pop().text
            self.updated_at = tree.findall("./bug/delta_ts").pop().text
            try:
                self.closed_at = (
                    tree.findall("./bug/cf_last_closed").pop().text
                )
            except IndexError:
                # cf_last_closed is present only if the issue has been closed
                # if not present it raises an IndexError, meaning the issue
                # isn't closed yet, which is a valid use case.
                pass