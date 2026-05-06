def add_env_var(self, key, value):
        """
        Add the given key / value as another environment
        variable.
        """
        self.manifest['env'][key] = value
        os.environ[key] = str(value)