def sourcess_list(self, *args):
        """Display a list of all registered events"""

        from pprint import pprint

        sources = {}
        sources.update(self.authorized_events)
        sources.update(self.anonymous_events)

        for source in sources:
            pprint(source)