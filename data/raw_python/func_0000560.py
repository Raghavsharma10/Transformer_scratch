def response_change(self, request, obj):
        """Determine the HttpResponse for the change_view stage."""
        opts = self.opts.app_label, self.opts.model_name
        pk_value = obj._get_pk_val()

        if '_continue' in request.POST:
            msg = _(
                'The %(name)s block was changed successfully. You may edit it again below.'
            ) % {'name': force_text(self.opts.verbose_name)}

            self.message_user(request, msg, messages.SUCCESS)

            # We redirect to the save and continue page, which updates the
            # parent window in javascript and redirects back to the edit page
            # in javascript.
            return HttpResponseRedirect(reverse(
                'admin:%s_%s_continue' % opts,
                args=(pk_value,),
                current_app=self.admin_site.name
            ))

        # Update column and close popup - don't bother with a message as they won't see it
        return self.response_rerender(request, obj, 'admin/glitter/update_column.html')