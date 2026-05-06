def plot(self, xmin, xmax, idx_input=0, idx_output=0, points=100,
             **kwargs) -> None:
        """Call method |anntools.ANN.plot| of all |anntools.ANN| objects
        handled by the actual |anntools.SeasonalANN| object.
        """
        for toy, ann_ in self:
            ann_.plot(xmin, xmax,
                      idx_input=idx_input, idx_output=idx_output,
                      points=points,
                      label=str(toy),
                      **kwargs)
        pyplot.legend()