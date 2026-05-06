def is_sufficient(self, device):
        """
        Returns whether the device is sufficient for this requirement.

        :param device: A GPUDevice instance.
        :type device: GPUDevice
        :return: True if the requirement is fulfilled otherwise False
        """

        sufficient = True
        if (self.min_vram is not None) and (device.vram < self.min_vram):
            sufficient = False

        return sufficient