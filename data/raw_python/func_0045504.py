def render(self, template, **data):
        """Renders the template using Jinja2 with given data arguments.

        """
        if(type(template) != str):
            raise TypeError("String expected")
        
        env = Environment(
            loader=FileSystemLoader(os.getcwd() + '/View'),
            autoescape=select_autoescape()
        )

        template = env.get_template(template)
        return self.finish(template.render(data))