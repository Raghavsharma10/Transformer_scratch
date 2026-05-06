def post_build(self, container_builder, container):
        """
        Register filter and global in jinja environment instance

        IoC tags are:
            - jinja2.filter to register filter, the tag must contain
            a name and a method options
            - jinja2.global to add new global, here globals are functions. 
            The tag must contain a name and a method options
        """
        jinja = container.get('ioc.extra.jinja2')

        for id in container_builder.get_ids_by_tag('jinja2.filter'):
            definition = container_builder.get(id)
            for option in definition.get_tag('jinja2.filter'):
                if 'name' not in option:
                    break

                if 'method' not in option:
                    break

                jinja.filters[option['name']] = getattr(container.get(id), option['method'])

        for id in container_builder.get_ids_by_tag('jinja2.global'):
            definition = container_builder.get(id)
            for option in definition.get_tag('jinja2.global'):

                if 'name' not in option:
                    break

                if 'method' not in option:
                    break                

                jinja.globals[option['name']] = getattr(container.get(id), option['method'])