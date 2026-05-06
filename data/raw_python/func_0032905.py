def run(self):
        """ Just copy the fixture, so we have some output. """
        luigi.LocalTarget(path=self.fixture).copy(self.output().path)