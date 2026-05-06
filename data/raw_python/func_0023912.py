def compare_view(self, request, object_id, version_id, extra_context=None):
        """Actually compare two versions."""
        opts = self.model._meta
        object_id = unquote(object_id)
        # get_for_object's ordering means this is always the latest revision.
        # The reversion we want to compare to
        current = Version.objects.get_for_object_reference(self.model, object_id)[0]
        revision = Version.objects.get_for_object_reference(self.model, object_id).filter(id=version_id)[0]

        the_diff = make_diff(current, revision)

        context = {
            "title": _("Comparing current %(model)s with revision created %(date)s") % {
                'model': current,
                'date' : get_date(revision),
                },
            "opts": opts,
            "compare_list_url": reverse("%s:%s_%s_comparelist" % (self.admin_site.name, opts.app_label, opts.model_name),
                                                                  args=(quote(object_id),)),
            "diff_list": the_diff,
        }

        extra_context = extra_context or {}
        context.update(extra_context)
        return render(request, self.compare_template or self._get_template_list("compare.html"),
                      context)