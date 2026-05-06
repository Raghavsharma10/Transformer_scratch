def log_action(self, instance, action, action_date=None, url="",
                   update_parent=True):
        """
        Store an action in the database using the CMSLog model.
        The following attributes are calculated and set on the log entry:

         * **model_repr** - A unicode representation of the instance.
         * **object_repr** - The verbose_name of the instance model class.
         * **section** - The name of ancestor bundle that is directly \
         attached to the admin site.

        :param instance: The instance that this action was performed \
        on.
        :param action: The action type. Must be one of the options \
        in CMSLog.ACTIONS.
        :param action_date: The datetime the action occurred.
        :param url: The url that the log entry should point to, \
        Defaults to an empty string.
        :param update_parent: If true this will update the last saved time \
        on the object pointed to by this bundle's object_view. \
        Defaults to True.
        """

        section = None
        if self.bundle:
            bundle = self.bundle
            while bundle.parent:
                bundle = bundle.parent
            section = bundle.name

        # if we have a object view that comes from somewhere else
        # save it too to update it.
        changed_object = instance
        bundle = self.bundle
        while bundle.object_view == bundle.parent_attr:
            bundle = bundle.parent

        if update_parent and changed_object.__class__ != bundle._meta.model:
            object_view, name = bundle.get_initialized_view_and_name(
                                    bundle.object_view, kwargs=self.kwargs)

            changed_object = object_view.get_object()
            changed_object.save()

        if not section:
            section = ""

        if url:
            url = urlparse.urlparse(url).path

        rep = unicode(instance)
        if rep:
            rep = rep[:255]

        log = CMSLog(action=action, url=url, section=section,
                     model_repr=instance._meta.verbose_name,
                     object_repr=rep,
                     user_name=self.request.user.username,
                     action_date=action_date)
        log.save()