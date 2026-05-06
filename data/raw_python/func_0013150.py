def get_all_kernel_specs_for_envs(self):
        """Returns the dict of name -> kernel_spec for all environments"""

        data = self._get_env_data()
        return {name: data[name][1] for name in data}