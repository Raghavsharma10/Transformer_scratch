def set_result(self, rval: bool) -> None:
        """ Set the result of the evaluation. If the result is true, prune all of the children that didn't cut it

        :param rval: Result of evaluation
        """
        self.result = rval
        if self.result:
            self.nodes = [pn for pn in self.nodes if pn.result]