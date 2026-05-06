def change_kernel(self, kernel):
        """
        Change the droplet's kernel

        :param kernel: a kernel ID or `Kernel` object representing the new
            kernel
        :type kernel: integer or `Kernel`
        :return: an `Action` representing the in-progress operation on the
            droplet
        :rtype: Action
        :raises DOAPIError: if the API endpoint replies with an error
        """
        if isinstance(kernel, Kernel):
            kernel = kernel.id
        return self.act(type='change_kernel', kernel=kernel)