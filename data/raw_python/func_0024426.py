def serialize(self):
        """
        Converts the form/model into JSON ready dicts/lists compatible
        with `Ulakbus-UI API`_.

        Example:

            .. code-block:: json

                {
                  "forms": {
                    "constraints": {},
                    "model": {
                      "code": null,
                      "name": null,
                      "save_edit": null,
                    },
                    "grouping": {},
                    "form": [
                      {
                        "helpvalue": null,
                        "type": "help"
                      },
                      "name",
                      "code",
                      "save_edit"
                    ],
                    "schema": {
                      "required": [
                        "name",
                        "code",
                        "save_edit"
                      ],
                      "type": "object",
                      "properties": {
                        "code": {
                          "type": "string",
                          "title": "Code Name"
                        },
                        "name": {
                          "type": "string",
                          "title": "Name"
                        },
                        "save_edit": {
                          "cmd": "save::add_edit_form",
                          "type": "button",
                          "title": "Save"
                        }
                      },
                      "title": "Add Permission"
                    }
                  }
                }
        """
        result = {
            "schema": {
                "title": self.title,
                "type": "object",
                "properties": {},
                "required": []
            },
            "form": [
                {
                    "type": "help",
                    "helpvalue": self.help_text
                }
            ],
            "model": {}
        }
        for itm in self.META_TO_FORM_ROOT:
            if itm in self.Meta.__dict__:
                result[itm] = self.Meta.__dict__[itm]

        if self._model.is_in_db():
            result["model"]['object_key'] = self._model.key
            result["model"]['model_type'] = self._model.__class__.__name__
            result["model"]['unicode'] = six.text_type(self._model)

        # if form intentionally marked as fillable from task data by assigning False to always_blank
        # field in Meta class, form_data is retrieved from task_data if exist in else None
        form_data = None
        if not self.Meta.always_blank:
            form_data = self.context.task_data.get(self.__class__.__name__, None)

        for itm in self._serialize():
            item_props = {'type': itm['type'], 'title': itm['title']}

            if not itm.get('value') and 'kwargs' in itm and 'value' in itm['kwargs']:
                itm['value'] = itm['kwargs'].pop('value')

            if 'kwargs' in itm and 'widget' in itm['kwargs']:
                item_props['widget'] = itm['kwargs'].pop('widget')

            if form_data:
                if form_data[itm['name']] and (itm['type'] == 'date' or itm['type'] == 'datetime'):
                    value_to_serialize = datetime.strptime(
                        form_data[itm['name']], itm['format'])
                else:
                    value_to_serialize = form_data[itm['name']]
                value = self._serialize_value(value_to_serialize)
                if itm['type'] == 'button':
                    value = None
            # if form_data is empty, value will be None, so it is needed to fill the form from model
            # or leave empty
            else:
                # if itm['value'] is not None returns itm['value']
                # else itm['default']
                if itm['value'] is not None:
                    value = itm['value']
                else:
                    value = itm['default']

            result["model"][itm['name']] = value

            if itm['type'] == 'model':
                item_props['model_name'] = itm['model_name']

            if itm['type'] not in ['ListNode', 'model', 'Node']:
                if 'hidden' in itm['kwargs']:
                    # we're simulating HTML's hidden form fields
                    # by just setting it in "model" dict and bypassing other parts
                    continue
                else:
                    item_props.update(itm['kwargs'])
            if itm.get('choices'):
                self._handle_choices(itm, item_props, result)
            else:
                result["form"].append(itm['name'])

            if 'help_text' in itm:
                item_props['help_text'] = itm['help_text']

            if 'schema' in itm:
                item_props['schema'] = itm['schema']

            # this adds default directives for building
            # add and list views of linked models
            if item_props['type'] == 'model':
                # this control for passing test.
                # object gets context but do not use it. why is it for?
                if self.context:
                    if self.context.has_permission("%s.select_list" % item_props['model_name']):
                        item_props.update({
                            'list_cmd': 'select_list',
                            'wf': 'crud',
                        })
                    if self.context.has_permission("%s.add_edit_form" % item_props['model_name']):
                        item_props.update({
                            'add_cmd': 'add_edit_form',
                            'wf': 'crud',
                        })
                else:
                    item_props.update({
                        'list_cmd': 'select_list',
                        'add_cmd': 'add_edit_form',
                        'wf': 'crud'
                    })
            result["schema"]["properties"][itm['name']] = item_props


            if itm['required']:
                result["schema"]["required"].append(itm['name'])
        self._cache_form_details(result)
        return result