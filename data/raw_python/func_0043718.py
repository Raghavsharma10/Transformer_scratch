def enumerate_sources(self):
        """ Return a list of (source_id, source_name) tuples """
        sources = []
        for source_id in range(1, 17):
            try:
                name = yield from self.get_source_variable(source_id, 'name')
                if name:
                    sources.append((source_id, name))
            except CommandException:
                break
        return sources