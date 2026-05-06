def return_dict(self):
        """Output dictionary for ``make_plot.py`` input.

        Iterates through the entire MainContainer class turning its contents
        into dictionary form. This dictionary becomes the input for ``make_plot.py``.

        If `print_input` attribute is True, the entire dictionary will be printed
        prior to returning the dicitonary.

        Returns:
            - **output_dict** (*dict*): Dicitonary for input into ``make_plot.py``.

        """
        output_dict = {}
        output_dict['general'] = self._iterate_through_class(self.general.__dict__)
        output_dict['figure'] = self._iterate_through_class(self.figure.__dict__)

        if self.total_plots > 1:
            trans_dict = ({
                           str(i): self._iterate_through_class(axis.__dict__) for i, axis
                          in enumerate(self.ax)})
            output_dict['plot_info'] = trans_dict

        else:
            output_dict['plot_info'] = {'0': self._iterate_through_class(self.ax.__dict__)}

        if self.print_input:
            print(output_dict)
        return output_dict