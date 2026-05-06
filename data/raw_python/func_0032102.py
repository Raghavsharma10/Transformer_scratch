def _render_context(self, template, block, **context):
        """
        Render a block to a string with its context
        """
        return u''.join(block(template.new_context(context)))