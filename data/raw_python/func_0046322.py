def returncode(self):
        '''
        Runs the command if it has not yet run (redirecting output to stdout).
        :return: The returncode of the last executed command in the chain.
        '''
        if not self._pop:
            self > sys.stdout
        link = self
        while link is not None:
            if link._pop:
                return link._pop.returncode
        return None