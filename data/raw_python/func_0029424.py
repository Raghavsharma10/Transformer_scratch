def add(self, member_name, collection_name='', parent=None, uid='',
            **kwargs):
        """
        :param member_name: singular name of the resource. It should be the
            appropriate singular version of the resource given your locale
            and used with members of the collection.

        :param collection_name: plural name of the resource. It will be used
            to refer to the resource collection methods and should be a
            plural version of the ``member_name`` argument.
            Note: if collection_name is empty, it means resource is singular

        :param parent: parent resource name or object.

        :param uid: unique name for the resource

        :param kwargs:
            view: custom view to overwrite the default one.
            the rest of the keyward arguments are passed to
            add_resource_routes call.

        :return: ResourceMap object
        """
        # self is the parent resource on which this method is called.
        parent = (self.resource_map.get(parent) if type(parent)
                  is str else parent or self)

        prefix = kwargs.pop('prefix', '')

        uid = (uid or
               ':'.join(filter(bool, [parent.uid, prefix, member_name])))

        if uid in self.resource_map:
            raise ValueError('%s already exists in resource map' % uid)

        # Use id_name of parent for singular views to make url generation
        # easier
        id_name = kwargs.get('id_name', '')
        if not id_name and parent:
            id_name = parent.id_name

        new_resource = Resource(self.config, member_name=member_name,
                                collection_name=collection_name,
                                parent=parent, uid=uid,
                                id_name=id_name,
                                prefix=prefix)

        view = maybe_dotted(
            kwargs.pop('view', None) or get_default_view_path(new_resource))

        for name, val in kwargs.pop('view_args', {}).items():
            setattr(view, name, val)

        root_resource = self.config.get_root_resource()

        view.root_resource = root_resource
        new_resource.view = view
        path_segs = []
        kwargs['path_prefix'] = ''

        for res in new_resource.ancestors:
            if not res.is_singular:
                if res.id_name:
                    id_full = res.id_name
                else:
                    id_full = "%s_%s" % (res.member_name, DEFAULT_ID_NAME)

                path_segs.append('%s/{%s}' % (res.collection_name, id_full))
            else:
                path_segs.append(res.member_name)

        if path_segs:
            kwargs['path_prefix'] = '/'.join(path_segs)

        if prefix:
            kwargs['path_prefix'] += '/' + prefix

        name_segs = [a.member_name for a in new_resource.ancestors]
        name_segs.insert(1, prefix)
        name_segs = [seg for seg in name_segs if seg]
        if name_segs:
            kwargs['name_prefix'] = '_'.join(name_segs) + ':'

        new_resource.renderer = kwargs.setdefault(
            'renderer', view._default_renderer)

        kwargs.setdefault('auth', root_resource.auth)
        kwargs.setdefault('factory', root_resource.default_factory)
        _factory = maybe_dotted(kwargs['factory'])

        kwargs['auth'] = kwargs.get('auth', root_resource.auth)

        kwargs['http_cache'] = kwargs.get(
            'http_cache', root_resource.http_cache)

        new_resource.action_route_map = add_resource_routes(
            self.config, view, member_name, collection_name,
            **kwargs)

        self.resource_map[uid] = new_resource
        # add all route names for this resource as keys in the dict,
        # so its easy to find it in the view.
        self.resource_map.update(dict.fromkeys(
            list(new_resource.action_route_map.values()),
            new_resource))

        # Store resources in {modelName: resource} map if:
        #   * Its view has Model defined
        #   * It's not singular
        #   * Its parent is root or it's not already stored
        model = new_resource.view.Model
        is_collection = model is not None and not new_resource.is_singular
        if is_collection:
            is_needed = (model.__name__ not in self.model_collections or
                         new_resource.parent is root_resource)
            if is_needed:
                self.model_collections[model.__name__] = new_resource

        parent.children.append(new_resource)
        view._resource = new_resource
        view._factory = _factory

        return new_resource