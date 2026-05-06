def return_dict(self):
        """Output dictionary for :mod:`gwsnrcalc.generate_contour_data` input.

        Iterates through the entire MainContainer class turning its contents
        into dictionary form. This dictionary becomes the input for
        :mod:`gwsnrcalc.generate_contour_data`.

        If `print_input` attribute is True, the entire dictionary will be printed
        prior to returning the dicitonary.

        Returns:
            - output_dict: Dicitonary for input into
                :mod:`gwsnrcalc.generate_contour_data`.

        """
        output_dict = {}
        output_dict['general'] = self._iterate_through_class(self.general.__dict__)
        output_dict['generate_info'] = self._iterate_through_class(self.generate_info.__dict__)
        output_dict['sensitivity_input'] = (self._iterate_through_class(
            self.sensitivity_input.__dict__))
        output_dict['snr_input'] = self._iterate_through_class(self.snr_input.__dict__)
        output_dict['parallel_input'] = self._iterate_through_class(self.parallel_input.__dict__)
        output_dict['output_info'] = self._iterate_through_class(self.output_info.__dict__)

        if self.print_input:
            print(output_dict)
        return output_dict