def random(self, size, n_and, max_in, n=1):
        """
        Generates `n` random logical networks with given size range, number of AND gates and maximum
        input signals for AND gates. Logical networks are saved in the attribute :attr:`networks`.

        Parameters
        ----------
        n : int
            Number of random logical networks to be generated

        size : (int,int)
            Minimum and maximum sizes

        n_and : (int,int)
            Minimum and maximum AND gates

        max_in : int
            Maximum input signals for AND gates
        """
        args = ['-c minsize=%s' % size[0], '-c maxsize=%s' % size[1],
                '-c minnand=%s' % n_and[0], '-c maxnand=%s' % n_and[1], '-c maxin=%s' % max_in]
        encodings = ['guess', 'random']

        self.networks.reset()

        clingo = self.__get_clingo__(args, encodings)
        clingo.conf.solve.models = str(n)
        clingo.conf.solver.seed = str(randint(0, 32767))
        clingo.conf.solver.sign_def = '3'

        clingo.ground([("base", [])])
        clingo.solve(on_model=self.__save__)