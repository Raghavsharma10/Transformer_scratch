def init_app(self, app):
        """Initialize Flask application."""
        prefix_len = len(self.prefix)
        for varname, value in os.environ.items():
            if not varname.startswith(self.prefix):
                continue

            # Prepare values
            varname = varname[prefix_len:]
            value = value or app.config.get(varname)

            # Evaluate value
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                pass

            # Set value
            app.config[varname] = value