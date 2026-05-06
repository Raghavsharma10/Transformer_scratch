def get_inline_instances(self, request, *args, **kwargs):
        """
        Create the inlines for the admin, including the placeholder and contentitem inlines.
        """
        inlines = super(PlaceholderEditorAdmin, self).get_inline_instances(request, *args, **kwargs)

        extra_inline_instances = []
        inlinetypes = self.get_extra_inlines()
        for InlineType in inlinetypes:
            inline_instance = InlineType(self.model, self.admin_site)
            extra_inline_instances.append(inline_instance)

        return extra_inline_instances + inlines