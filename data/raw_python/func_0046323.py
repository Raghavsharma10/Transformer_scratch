def reset(self):
        '''
        Allows you to re-run the command chain.
        :return: self
        '''
        t = self
        while t._output is not None:
            t = t._output
        while t is not None:
            if t._pop and t._pop.returncode is None:
                t._pop.kill()
                t._pop.wait()
            del t._pop
            t._pop = None
            t = t._input
        return self