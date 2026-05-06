def run(self):
        """ Just run wget quietly. """
        output = shellout('wget -q "{url}" -O {output}', url=self.url)
        luigi.LocalTarget(output).move(self.output().path)