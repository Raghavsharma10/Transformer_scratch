def add_noise_curve(self, name, noise_type='ASD', is_wd_background=False):
        """Add a noise curve for generation.

        This will add a noise curve for an SNR calculation by appending to the sensitivity_curves
        list within the sensitivity_input dictionary.

        The name of the noise curve prior to the file extension will appear as its
        label in the final output dataset. Therefore, it is recommended prior to
        running the generator that file names are renamed to simple names
        for later reference.

        Args:
            name (str): Name of noise curve including file extension inside input_folder.
            noise_type (str, optional): Type of noise. Choices are `ASD`, `PSD`, or `char_strain`.
                Default is ASD.
            is_wd_background (bool, optional): If True, this sensitivity is used as the white dwarf
                background noise. Default is False.

        """
        if is_wd_background:
            self.sensitivity_input.wd_noise = name
            self.sensitivity_input.wd_noise_type_in = noise_type

        else:
            if 'sensitivity_curves' not in self.sensitivity_input.__dict__:
                self.sensitivity_input.sensitivity_curves = []
            if 'noise_type_in' not in self.sensitivity_input.__dict__:
                self.sensitivity_input.noise_type_in = []

            self.sensitivity_input.sensitivity_curves.append(name)
            self.sensitivity_input.noise_type_in.append(noise_type)
        return