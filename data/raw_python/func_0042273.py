def get_view_and_name(self, attname):
        """
        Gets a view or bundle and returns it
        and it's url_name.
        """
        view = getattr(self, attname, None)
        if attname in self._children:
            view = self._get_bundle_from_promise(attname)

        if view:
            if attname in self._children:
                return view, view.name
            elif isinstance(view, ViewAlias):
                view_name = view.get_view_name(attname)
                bundle = view.get_bundle(self, {}, {})
                if bundle and isinstance(bundle, Bundle):
                    view, name = bundle.get_view_and_name(view_name)

            if hasattr(view, 'as_view'):
                if attname != 'main':
                    name = "%s_%s" % (self.name, attname)
                else:
                    name = self.name
                return view, name
            elif view == self.parent_attr and self.parent:
                return self.parent_attr, None
            elif isinstance(view, URLAlias):
                return view, None

        return None, None