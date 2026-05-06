def render_blocks(self, template_name, **context):
        """
        To render all the blocks
        :param template_name: The template file name
        :param context: **kwargs context to render
        :retuns dict: of all the blocks with block_name as key
        """
        blocks = {}
        template = self._get_template(template_name)
        for block in template.blocks:
            blocks[block] = self._render_context(template,
                                                template.blocks[block],
                                                **context)
        return blocks