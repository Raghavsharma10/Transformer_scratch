def terminate_process(self, idf):
        """ Terminate a process by id """
        try:
            p = self.q.pop(idf)
            p.terminate()
            return p
        except:
            return None