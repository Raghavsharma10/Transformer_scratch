def generate(self, labels, split_idx):
        """Generate peak-specific noise abstract method, must be reimplemented in a subclass.

        :param tuple labels: Dimension labels of a peak.
        :param int split_idx: Index specifying which peak list split parameters to use.
        :return: List of noise values for dimensions ordered as they appear in a peak.
        :rtype: :py:class:`list`
        """
        atom_labels = [label[0] for label in labels]

        noise = []
        distribution_function = distributions[self.distribution_name]["function"]
        for label in atom_labels:
            params = [self.parameters["{}_{}".format(label, param)][split_idx]
                      for param in self.distribution_parameter_names]

            if None in params:
                dim_noise = 0.0
            else:
                try:
                    dim_noise = distribution_function(*params)
                except ValueError:
                    raise ValueError

            noise.append(dim_noise)

        return noise