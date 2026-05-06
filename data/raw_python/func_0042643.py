def form_valid(self, form, formsets):
        """
        Response for valid form. In one transaction this will
        save the current form and formsets, log the action
        and message the user.

        Returns the results of calling the `success_response` method.
        """
        # check if it's a new object before it save the form
        new_object = False
        if not self.object:
            new_object = True

        instance = getattr(form, 'instance', None)
        auto_tags, changed_tags, old_tags = tag_handler.get_tags_from_data(
            form.data, self.get_tags(instance))
        tag_handler.set_auto_tags_for_form(form, auto_tags)

        with transaction.commit_on_success():
            self.object = self.save_form(form)
            self.save_formsets(form, formsets, auto_tags=auto_tags)

            url = self.get_object_url()
            self.log_action(self.object, CMSLog.SAVE, url=url)
            msg = self.write_message()

        # get old and new tags
        if not new_object and changed_tags and old_tags:
            tag_handler.update_changed_tags(changed_tags, old_tags)

        return self.success_response(msg)