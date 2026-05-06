def find_kernel_specs(self):
        """Returns a dict mapping kernel names to resource directories."""
        # let real installed kernels overwrite envs with the same name:
        # this is the same order as the get_kernel_spec way, which also prefers
        # kernels from the jupyter dir over env kernels.
        specs = self.find_kernel_specs_for_envs()
        specs.update(super(EnvironmentKernelSpecManager,
                           self).find_kernel_specs())
        return specs