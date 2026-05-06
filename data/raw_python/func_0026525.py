def search(self, event):
        """Search for an object"""

        try:
            data, schema, user, client = self._get_args(event)
        except AttributeError:
            return

        # object_filter['$text'] = {'$search': str(data['search'])}
        if data.get('fulltext', False) is True:
            object_filter = {
                'name': {
                    '$regex': str(data['search']),
                    '$options': '$i'
                }
            }
        else:
            if isinstance(data['search'], dict):
                object_filter = data['search']
            else:
                object_filter = {}

        if 'fields' in data:
            fields = data['fields']
        else:
            fields = []

        skip = data.get('skip', 0)
        limit = data.get('limit', 0)
        sort = data.get('sort', None)
        # page = data.get('page', 0)
        # count = data.get('count', 0)
        #
        # if page > 0 and count > 0:
        #     skip = page * count
        #     limit = count

        if 'subscribe' in data:
            self.log('Subscription:', data['subscribe'], lvl=verbose)
            do_subscribe = data['subscribe'] is True
        else:
            do_subscribe = False

        object_list = []

        size = objectmodels[schema].count(object_filter)

        if size > WARNSIZE and (limit > 0 and limit > WARNSIZE):
            self.log("Getting a very long (", size, ") list of items for ", schema,
                     lvl=warn)

        opts = schemastore[schema].get('options', {})
        hidden = opts.get('hidden', [])

        self.log("object_filter: ", object_filter, ' Schema: ', schema,
                 "Fields: ", fields,
                 lvl=verbose)

        options = {}
        if skip > 0:
            options['skip'] = skip
        if limit > 0:
            options['limit'] = limit
        if sort is not None:
            options['sort'] = []
            for item in sort:
                key = item[0]
                direction = item[1]
                direction = ASCENDING if direction == 'asc' else DESCENDING
                options['sort'].append([key, direction])

        cursor = objectmodels[schema].find(object_filter, **options)

        for item in cursor:
            if not self._check_permissions(user, 'list', item):
                continue
            self.log("Search found item: ", item, lvl=verbose)

            try:
                list_item = {'uuid': item.uuid}
                if fields in ('*', ['*']):
                    item_fields = item.serializablefields()
                    for field in hidden:
                        item_fields.pop(field, None)
                    object_list.append(item_fields)
                else:
                    if 'name' in item._fields:
                        list_item['name'] = item.name

                    for field in fields:
                        if field in item._fields and field not in hidden:
                            list_item[field] = item._fields[field]
                        else:
                            list_item[field] = None

                    object_list.append(list_item)

                if do_subscribe:
                    self._add_subscription(item.uuid, event)
            except Exception as e:
                self.log("Faulty object or field: ", e, type(e),
                         item._fields, fields, lvl=error, exc=True)
        # self.log("Generated object search list: ", object_list)

        result = {
            'component': 'hfos.events.objectmanager',
            'action': 'search',
            'data': {
                'schema': schema,
                'list': object_list,
                'size': size
            }
        }

        self._respond(None, result, event)