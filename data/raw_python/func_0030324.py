def render(self, context):
        """
        Render the tag, with extra context layer.
        """
        extra_context = self.context_expr.resolve(context)
        if not isinstance(extra_context, dict):
            raise TemplateSyntaxError("{% withdict %} expects the argument to be a dictionary.")

        with context.push(**extra_context):
            return self.nodelist.render(context)