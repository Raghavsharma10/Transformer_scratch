def terminate(self):
        """ Terminate all processes """
        for w in self.q.values():
            try:
                w.terminate()
            except:
                pass

        self.q = {}