def setup_function(self):
        """twitter.comon.app runs this before any global main() function."""
        fpm_path = app.get_options().fpm_bin
        if not os.path.exists(fpm_path):
            log.warn('Could not find fpm; gendeb cannot function.')
        else:
            GenDebBuilder.fpm_bin = fpm_path