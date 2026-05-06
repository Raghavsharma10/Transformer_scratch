def plot(self, xmin, xmax, idx_input=0, idx_output=0, points=100,
             **kwargs) -> None:
        """Plot the relationship between a certain input (`idx_input`) and a
        certain output (`idx_output`) variable described by the actual
        |anntools.ANN| object.

        Define the lower and the upper bound of the x axis via arguments
        `xmin` and `xmax`.  The number of plotting points can be modified
        by argument `points`.  Additional `matplotlib` plotting arguments
        can be passed as keyword arguments.
        """
        xs_ = numpy.linspace(xmin, xmax, points)
        ys_ = numpy.zeros(xs_.shape)
        for idx, x__ in enumerate(xs_):
            self.inputs[idx_input] = x__
            self.process_actual_input()
            ys_[idx] = self.outputs[idx_output]
        pyplot.plot(xs_, ys_, **kwargs)