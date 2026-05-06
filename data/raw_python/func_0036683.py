def alive(self):
        '''Is this component alive?'''
        with self._mutex:
            if self.exec_contexts:
                for ec in self.exec_contexts:
                    if self._obj.is_alive(ec):
                        return True
        return False