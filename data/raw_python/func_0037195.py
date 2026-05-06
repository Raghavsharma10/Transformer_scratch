def apply_noise(self, noise_generator, split_idx, ndigits=6):
        """Apply noise to dimensions within a peak.

        :param noise_generator: Noise generator object.
        :param int split_idx: Index specifying which peak list split parameters to use.
        :return: None
        :rtype: :py:obj:`None`
        """
        noise = noise_generator.generate(self.labels, split_idx)
        for dim, noise_value in zip(self, noise):
            dim.chemshift = round(dim.chemshift + noise_value, ndigits)