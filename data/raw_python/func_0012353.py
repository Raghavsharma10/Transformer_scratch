def add(self, snapshot, component='main'):
        """
        Add snapshot of component to publish
        """
        try:
            self.components[component].append(snapshot)
        except KeyError:
            self.components[component] = [snapshot]