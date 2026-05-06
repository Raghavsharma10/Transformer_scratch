def delete(self, request, *args, **kwargs):
        """Override delete to only withdraw"""
        talk = self.get_object()
        talk.status = WITHDRAWN
        talk.save()
        revisions.set_user(self.request.user)
        revisions.set_comment("Talk Withdrawn")
        return HttpResponseRedirect(self.success_url)