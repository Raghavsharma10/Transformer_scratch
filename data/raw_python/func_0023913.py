def comparelist_view(self, request, object_id, extra_context=None):
        """Allow selecting versions to compare."""
        opts = self.model._meta
        object_id = unquote(object_id)
        current = get_object_or_404(self.model, pk=object_id)
        # As done by reversion's history_view
        action_list = [
            {
                "revision": version.revision,
                "url": reverse("%s:%s_%s_compare" % (self.admin_site.name, opts.app_label, opts.model_name), args=(quote(version.object_id), version.id)),
            } for version in self._reversion_order_version_queryset(Version.objects.get_for_object_reference(
                self.model,
                object_id).select_related("revision__user"))]
        context = {"action_list": action_list,
                   "opts": opts,
                   "object_id": quote(object_id),
                   "original": current,
                  }
        extra_context = extra_context or {}
        context.update(extra_context)
        return render(request, self.compare_list_template or self._get_template_list("compare_list.html"),
                      context)