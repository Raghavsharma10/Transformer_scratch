def objectlist(self, event):
        """Get a list of objects"""

        self.log('LEGACY LIST FUNCTION CALLED!', lvl=warn)
        try:
            data, schema, user, client = self._get_args(event)
        except AttributeError:
            return

        object_filter = self._get_filter(event)
        self.log('Object list for', schema, 'requested from',
                 user.account.name, lvl=debug)

        if 'fields' in data:
            fields = data['fields']
        else:
            fields = []

        object_list = []

        opts = schemastore[schema].get('options', {})
        hidden = opts.get('hidden', [])

        if objectmodels[schema].count(object_filter) > WARNSIZE:
            self.log("Getting a very long list of items for ", schema,
                     lvl=warn)

        try:
            for item in objectmodels[schema].find(object_filter):
                try:
                    if not self._check_permissions(user, 'list', item):
                        continue
                    if fields in ('*', ['*']):
                        item_fields = item.serializablefields()
                        for field in hidden:
                            item_fields.pop(field, None)
                        object_list.append(item_fields)
                    else:
                        list_item = {'uuid': item.uuid}

                        if 'name' in item._fields:
                            list_item['name'] = item._fields['name']

                        for field in fields:
                            if field in item._fields and field not in hidden:
                                list_item[field] = item._fields[field]
                            else:
                                list_item[field] = None

                        object_list.append(list_item)
                except Exception as e:
                    self.log("Faulty object or field: ", e, type(e),
                             item._fields, fields, lvl=error, exc=True)
        except ValidationError as e:
            self.log('Invalid object in database encountered!', e, exc=True,
                     lvl=warn)
        # self.log("Generated object list: ", object_list)

        result = {
            'component': 'hfos.events.objectmanager',
            'action': 'getlist',
            'data': {
                'schema': schema,
                'list': object_list
            }
        }

        self._respond(None, result, event)