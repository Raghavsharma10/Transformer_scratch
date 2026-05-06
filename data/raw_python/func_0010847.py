def view_for_action(self, action):
        """
        Returns the appropriate view class for the passed in action
        """
        # this turns replace_foo into ReplaceFoo and read into Read
        class_name = "".join([word.capitalize() for word in action.split("_")])
        view = None

        # see if we have a custom class defined for this action
        if hasattr(self, class_name):
            # return that one
            view = getattr(self, class_name)

            # no model set?  set it ourselves
            if not getattr(view, 'model', None):
                view.model = self.model

            # no permission and we are supposed to set them, do so
            if not hasattr(view, 'permission') and self.permissions:
                view.permission = self.permission_for_action(action)

            # set our link URL based on read and update
            if not getattr(view, 'link_url', None):
                if 'read' in self.actions:
                    view.link_url = 'id@%s' % self.url_name_for_action('read')
                elif 'update' in self.actions:
                    view.link_url = 'id@%s' % self.url_name_for_action('update')

            # if we can't infer a link URL then view class must override lookup_field_link
            if not getattr(view, 'link_url', None) and 'lookup_field_link' not in view.__dict__:
                view.link_fields = ()

            # set add_button based on existence of Create view if add_button not explicitly set
            if action == 'list' and getattr(view, 'add_button', None) is None:
                view.add_button = 'create' in self.actions

            # set edit_button based on existence of Update view if edit_button not explicitly set
            if action == 'read' and getattr(view, 'edit_button', None) is None:
                view.edit_button = 'update' in self.actions

            # if update or create, set success url if not set
            if not getattr(view, 'success_url', None) and (action == 'update' or action == 'create'):
                view.success_url = '@%s' % self.url_name_for_action('list')

        # otherwise, use our defaults
        else:
            options = dict(model=self.model)

            # if this is an update or create, and we have a list view, then set the default to that
            if action == 'update' or action == 'create' and 'list' in self.actions:
                options['success_url'] = '@%s' % self.url_name_for_action('list')

            # set permissions if appropriate
            if self.permissions:
                options['permission'] = self.permission_for_action(action)

            if action == 'create':
                view = type(str("%sCreateView" % self.model_name), (SmartCreateView,), options)

            elif action == 'read':
                if 'update' in self.actions:
                    options['edit_button'] = True

                view = type(str("%sReadView" % self.model_name), (SmartReadView,), options)

            elif action == 'update':
                if 'delete' in self.actions:
                    options['delete_url'] = 'id@%s' % self.url_name_for_action('delete')

                view = type(str("%sUpdateView" % self.model_name), (SmartUpdateView,), options)

            elif action == 'delete':
                if 'list' in self.actions:
                    options['cancel_url'] = '@%s' % self.url_name_for_action('list')
                    options['redirect_url'] = '@%s' % self.url_name_for_action('list')

                elif 'update' in self.actions:
                    options['cancel_url'] = '@%s' % self.url_name_for_action('update')

                view = type(str("%sDeleteView" % self.model_name), (SmartDeleteView,), options)

            elif action == 'list':
                if 'read' in self.actions:
                    options['link_url'] = 'id@%s' % self.url_name_for_action('read')
                elif 'update' in self.actions:
                    options['link_url'] = 'id@%s' % self.url_name_for_action('update')
                else:
                    options['link_fields'] = ()

                if 'create' in self.actions:
                    options['add_button'] = True

                view = type(str("%sListView" % self.model_name), (SmartListView,), options)

            elif action == 'csv_import':
                options['model'] = ImportTask
                view = type(str("%sCSVImportView" % self.model_name), (SmartCSVImportView,), options)

        if not view:
            # couldn't find a view?  blow up
            raise Exception("No view found for action: %s" % action)

        # set the url name for this view
        view.url_name = self.url_name_for_action(action)

        # no template set for it?  set one based on our action and app name
        if not getattr(view, 'template_name', None):
            view.template_name = self.template_for_action(action)

        view.crudl = self

        return view