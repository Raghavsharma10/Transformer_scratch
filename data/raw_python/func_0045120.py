def make_frame(self, frame, birthframe, startframe, stopframe, deathframe, noiseframe=None):
        """
        animation happens between startframe and stopframe
        the value is None before aliveframe, and after deathframe
         * if aliveframe is not specified it defaults to startframe
         * if deathframe is not specified it defaults to stopframe

        initial value is held from aliveframe to startframe

        final value is held from stopfrome to deathframe 
        """

        if birthframe is None:
            birthframe = startframe
        if deathframe is None:
            deathframe = stopframe
        if frame < birthframe:
            return None
        if frame > deathframe:
            return None
        if frame < startframe:
            return self.frm
        if frame > stopframe:
            return self.to

        parameter_value = self.T.tween2(frame, startframe, stopframe)
        t = Symbol('t')
        if self.noise_fn is not None:
            if noiseframe is not None:
                nf = noiseframe
            else:
                nf = parameter_value
            noise_value = self.noise_fn(frame, nf)
        else:
            noise_value = 0
        return self.equation.evalf(subs={t: parameter_value}) + noise_value