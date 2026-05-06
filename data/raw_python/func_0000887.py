def serialize(self):
        """Serialize into JSONable dict, and associated locations data."""
        api_metadata = OrderedDict()
        # $ char makes this come first in sort ordering
        api_metadata['$version'] = self.current_version
        locations = {}

        for svc_name, group in self.groups():
            group_apis = OrderedDict()
            group_metadata = OrderedDict()
            group_metadata['apis'] = group_apis
            group_metadata['title'] = group.title
            api_metadata[group.name] = group_metadata

            if group.docs is not None:
                group_metadata['docs'] = group.docs

            for name, api in group.items():
                group_apis[name] = OrderedDict()
                group_apis[name]['service'] = svc_name
                group_apis[name]['api_group'] = group.name
                group_apis[name]['api_name'] = api.name
                group_apis[name]['introduced_at'] = api.introduced_at
                group_apis[name]['methods'] = api.methods
                group_apis[name]['request_schema'] = api.request_schema
                group_apis[name]['response_schema'] = api.response_schema
                group_apis[name]['doc'] = api.docs
                group_apis[name]['changelog'] = api._changelog
                if api.title:
                    group_apis[name]['title'] = api.title
                else:
                    title = name.replace('-', ' ').replace('_', ' ').title()
                    group_apis[name]['title'] = title

                group_apis[name]['url'] = api.resolve_url()

                if api.undocumented:
                    group_apis[name]['undocumented'] = True
                if api.deprecated_at is not None:
                    group_apis[name]['deprecated_at'] = api.deprecated_at

                locations[name] = {
                    'api': api.location,
                    'request_schema': api._request_schema_location,
                    'response_schema': api._response_schema_location,
                    'changelog': api._changelog_locations,
                    'view': api.view_fn_location,
                }

        return api_metadata, locations