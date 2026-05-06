def verify(self):
        """
        Verifying all inspectors in exp_list
        Return:
            True: pass all inspectors
            False: fail at more than one inspector
        """
        for expectation in self.exp_list:
            if hasattr(expectation, "verify") and not expectation.verify():
                return False
        return True