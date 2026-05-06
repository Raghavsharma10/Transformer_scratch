def find_kernel_specs_for_envs(self):
        """Returns a dict mapping kernel names to resource directories."""
        data = self._get_env_data()
        return {name: data[name][0] for name in data}