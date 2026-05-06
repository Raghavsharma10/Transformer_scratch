def post_build(self, container_builder, container):
        """
        This method make sure the flask configuration is fine, and 
        check the if ioc.extra.jinja2 service is available. If so, the 
        flask instance will use this service, by keeping the flask template 
        loader and the one registered at the jinja2
        """
        app = container.get('ioc.extra.flask.app')

        app.config.update(container_builder.parameters.get('ioc.extra.flask.app.config'))

        if container.has('ioc.extra.jinja2'):
            # This must be an instance of jinja.ChoiceLoader
            # This code replace the flask specific jinja configuration to use
            # the one provided by the ioc.extra.jinja2 code
            jinja2 = container.get('ioc.extra.jinja2')

            jinja2.loader.loaders.append(app.create_global_jinja_loader())

            for name, value in app.jinja_env.globals.items():
                if name not in jinja2.globals:
                    jinja2.globals[name] = value                

            for name, value in app.jinja_env.filters.items():
                if name not in jinja2.filters:
                    jinja2.filters[name] = value                

            app.jinja_env = jinja2