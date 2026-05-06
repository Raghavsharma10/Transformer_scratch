def events_list(self, *args):
        """Display a list of all registered events"""

        def merge(a, b, path=None):
            "merges b into a"
            if path is None: path = []
            for key in b:
                if key in a:
                    if isinstance(a[key], dict) and isinstance(b[key], dict):
                        merge(a[key], b[key], path + [str(key)])
                    elif a[key] == b[key]:
                        pass  # same leaf value
                    else:
                        raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
                else:
                    a[key] = b[key]
            return a

        events = {}
        sources = merge(self.authorized_events, self.anonymous_events)

        for source, source_events in sources.items():
            events[source] = []
            for item in source_events:
                events[source].append(item)

        self.log(events, pretty=True)