def set_env_var(key: str, value: str):
        """
        Sets environment variable on AV

        Args:
            key: variable name
            value: variable value
        """
        elib_run.run(f'appveyor SetVariable -Name {key} -Value {value}')
        AV.info('Env', f'set "{key}" -> "{value}"')