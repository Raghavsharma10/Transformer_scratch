def change_view(self, request, object_id, form_url='', extra_context=None):
        """The 'change' admin view for this model."""

        obj = self.get_object(request, unquote(object_id))

        if obj is None:
            raise Http404(_('%(name)s object with primary key %(key)r does not exist.') % {
                'name': force_text(self.opts.verbose_name),
                'key': escape(object_id),
            })

        if not self.has_change_permission(request, obj):
            raise PermissionDenied

        content_block = obj.content_block
        version = content_block.obj_version

        # Version must not be saved, and must belong to this user
        if version.version_number or version.owner != request.user:
            raise PermissionDenied

        return super().change_view(request, object_id, form_url, extra_context)