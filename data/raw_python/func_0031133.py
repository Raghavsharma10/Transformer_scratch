def get_fixture_node(self, app_label, fixture_prefix):
        """
        Get all fixtures in given app with given prefix.
        :param str app_label: App label
        :param str fixture_prefix: first part of the fixture name
        :return: list of found fixtures.
        """
        app_nodes = self.get_app_nodes(app_label=app_label)
        nodes = [
            node for node in app_nodes if node[1].startswith(fixture_prefix)
            ]

        if len(nodes) > 1:
            raise MultipleFixturesFound(
                "The following fixtures with prefix '%s' are found in app '%s'"
                ": %s" % (
                    fixture_prefix, app_label, ', '.join(
                        [node[1] for node in nodes]
                    )
                )
            )
        elif len(nodes) == 0:
            raise FixtureNotFound("Fixture with prefix '%s' not found in app "
                                  "'%s'" % (fixture_prefix, app_label))
        return nodes