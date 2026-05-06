def history_view(self, request, object_id, extra_context=None):
        from django.template.response import TemplateResponse
        from django.contrib.admin.options import get_content_type_for_model
        from django.contrib.admin.utils import unquote
        from django.core.exceptions import PermissionDenied
        from django.utils.text import capfirst
        from django.utils.encoding import force_text
        from django.utils.translation import ugettext as _

        "The 'history' admin view for this model."
        from django.contrib.admin.models import LogEntry
        # First check if the user can see this history.
        model = self.model
        obj = self.get_object(request, unquote(object_id))
        if obj is None:
            return self._get_obj_does_not_exist_redirect(request, model._meta, object_id)

        if not self.has_change_permission(request, obj):
            raise PermissionDenied

        # Then get the history for this object.
        opts = model._meta
        app_label = opts.app_label
        action_list = LogEntry.objects.filter(
            object_id=unquote(object_id),
            content_type=get_content_type_for_model(model)
        ).select_related().order_by('-action_time')[:self.max_history_length]

        context = dict(
            self.admin_site.each_context(request),
            title=_('Change history: %s') % force_text(obj),
            action_list=action_list,
            module_name=capfirst(force_text(opts.verbose_name_plural)),
            object=obj,
            opts=opts,
            preserved_filters=self.get_preserved_filters(request),
        )
        context.update(extra_context or {})

        request.current_app = self.admin_site.name

        return TemplateResponse(request, self.object_history_template or [
            "admin/%s/%s/object_history.html" % (app_label, opts.model_name),
            "admin/%s/object_history.html" % app_label,
            "admin/object_history.html"
        ], context)