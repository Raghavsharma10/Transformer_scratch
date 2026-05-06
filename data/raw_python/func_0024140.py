def iter_size_changes(self, issue):
        """Yield an IssueSnapshot for each time the issue size changed
        """

        # Find the first size change, if any
        try:
            size_changes = list(filter(lambda h: h.field == 'Story Points',
                                       itertools.chain.from_iterable([c.items for c in issue.changelog.histories])))
        except AttributeError:
            return

        # If we have no size changes and the issue has a current size then a size must have ben specified at issue creation time.
        # Return the size at creation time

        try:
            current_size = issue.fields.__dict__[self.fields['StoryPoints']]
        except:
            current_size = None

        size = (size_changes[0].fromString) if len(size_changes)  else current_size

        # Issue was created
        yield IssueSizeSnapshot(
            change=None,
            key=issue.key,
            date=dateutil.parser.parse(issue.fields.created),
            size=size
        )

        for change in issue.changelog.histories:
            change_date = dateutil.parser.parse(change.created)

            #sizes = list(filter(lambda i: i.field == 'Story Points', change.items))
            #is_resolved = (sizes[-1].to is not None) if len(sizes) > 0 else is_resolved

            for item in change.items:
                if item.field == 'Story Points':
                    # StoryPoints value was changed
                    size = item.toString
                    yield IssueSizeSnapshot(
                        change=item.field,
                        key=issue.key,
                        date=change_date,
                        size=size
                    )