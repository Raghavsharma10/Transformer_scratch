def change_view(self, request, object_id, form_url='', extra_context=None):
        """
        Override change view to add extra context enabling moderate tool.
        """
        context = {
            'has_moderate_tool': True
        }
        if extra_context:
            context.update(extra_context)
        return super(AdminModeratorMixin, self).change_view(
            request=request,
            object_id=object_id,
            form_url=form_url,
            extra_context=context
        )