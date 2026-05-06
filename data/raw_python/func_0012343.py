def add(self, snapshot, distributions, component='main', storage=""):
        """ Add mirror or repo to publish """
        for dist in distributions:
            self.publish(dist, storage=storage).add(snapshot, component)