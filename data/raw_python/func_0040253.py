def setup_function(self):
        """Runs prior to the global main function."""
        log.options.LogOptions.set_stderr_log_level('google:INFO')
        if app.get_options().debug:
            log.options.LogOptions.set_stderr_log_level('google:DEBUG')
        if not app.get_options().build_root:
            app.set_option('build_root', os.path.join(
                app.get_options().butcher_basedir, 'build'))
        self.buildroot = app.get_options().build_root
        if not os.path.exists(self.buildroot):
            os.makedirs(self.buildroot)
        if app.get_options().disable_cache_fetch:
            self.options['cache_fetch'] = False
        if app.get_options().disable_hardlinks:
            base.BaseBuilder.linkfiles = False