def get_kernel_spec(self, kernel_name):
        """Returns a :class:`KernelSpec` instance for the given kernel_name.

        Raises :exc:`NoSuchKernel` if the given kernel name is not found.
        """
        try:
            return super(EnvironmentKernelSpecManager,
                         self).get_kernel_spec(kernel_name)
        except (NoSuchKernel, FileNotFoundError):
            venv_kernel_name = kernel_name.lower()
            specs = self.get_all_kernel_specs_for_envs()
            if venv_kernel_name in specs:
                return specs[venv_kernel_name]
            else:
                raise NoSuchKernel(kernel_name)