def render(self, template_name, block, **context):
        """
        TO render a block in the template
        :param template_name: the template file name
        :param block: the name of the block within {% block $block_name %}
        :param context: **kwargs context to render
        :returns string: of rendered content
        """
        template = self._get_template(template_name)
        return self._render_context(template,
                                    template.blocks[block],
                                    **context)