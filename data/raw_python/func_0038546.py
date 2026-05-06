def jinja_env(self):
        """Create a sandboxed Jinja environment."""
        if self._jinja_env is None:
            self._jinja_env = SandboxedEnvironment(
                extensions=[
                    'jinja2.ext.autoescape', 'jinja2.ext.with_', ],
                autoescape=True,
            )
            self._jinja_env.globals['url_for'] = url_for
            # Load whitelisted configuration variables.
            for var in self.app.config['PAGES_WHITELIST_CONFIG_KEYS']:
                self._jinja_env.globals[var] = self.app.config.get(var)
        return self._jinja_env