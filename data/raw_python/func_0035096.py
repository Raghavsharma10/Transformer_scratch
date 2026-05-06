def savefig(self, output_path, **kwargs):
        """Save figure during generation.

        This method is used to save a completed figure during the main function run.
        It represents a call to ``matplotlib.pyplot.fig.savefig``.

        # TODO: Switch to kwargs for matplotlib.pyplot.savefig

        Args:
            output_path (str): Relative path to the WORKING_DIRECTORY to save the figure.

        Keyword Arguments:
            dpi (int, optional): Dots per inch of figure. Default is 200.
            Note: Other kwargs are available. See:
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html

        """
        self.figure.save_figure = True
        self.figure.output_path = output_path
        self.figure.savefig_kwargs = kwargs
        return