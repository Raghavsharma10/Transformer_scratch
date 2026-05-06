def inject(self, seed=None, urlDir=None, **args):
        """
        :param seed: A Seed object (this or urlDir must be specified)
        :param urlDir: The directory on the server containing the seed list (this or urlDir must be specified)
        :param args: Extra arguments for the job
        :return: a created Job object
        """

        if seed:
            if urlDir and urlDir != seed.seedPath:
                raise NutchException("Can't specify both seed and urlDir")
            urlDir = seed.seedPath
        elif urlDir:
            pass
        else:
            raise NutchException("Must specify seed or urlDir")
        args['url_dir'] = urlDir
        return self.create('INJECT', **args)