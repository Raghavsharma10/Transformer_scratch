def get_all_specs(self):
        """Returns a dict mapping kernel names and resource directories.
        """
        # This is new in 4.1 -> https://github.com/jupyter/jupyter_client/pull/93
        specs = self.get_all_kernel_specs_for_envs()
        specs.update(super(EnvironmentKernelSpecManager, self).get_all_specs())
        return specs