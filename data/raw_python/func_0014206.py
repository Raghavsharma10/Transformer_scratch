def template_scripts(self, config, template_name):
        '''
        Returns a list of scripts used by the given template object AND its ancestors.

        This runs a ProviderRun on the given template (as if it were being displayed).
        This allows the WEBPACK_PROVIDERS to provide the JS files to us.
        '''
        dmp = apps.get_app_config('django_mako_plus')
        template_obj = dmp.engine.get_template_loader(config, create=True).get_mako_template(template_name, force=True)
        mako_context = create_mako_context(template_obj)
        inner_run = WebpackProviderRun(mako_context['self'])
        inner_run.run()
        scripts = []
        for tpl in inner_run.templates:
            for p in tpl.providers:
                if os.path.exists(p.absfilepath):
                    scripts.append(p.absfilepath)
        return scripts