def paramsReport(self):
        """See docs for `Model` abstract base class."""
        report = self._models[0].paramsReport
        del report[self.distributedparam]
        for param in self.distributionparams:
            new_name = "_".join([param.split("_")[0], self.distributedparam])
            report[new_name] = getattr(self, param)
        return report