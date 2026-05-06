def generate(self, groups='all', sort=None):
        """Return a list of dict describing the routes specified by the
        doc() method

        Each dict contains:
         - methods: the set of allowed methods (ie ['GET', 'POST'])
         - rule: relative url (ie '/user/<int:id>')
         - endpoint: function name (ie 'show_user')
         - doc: docstring of the function
         - args: function arguments
         - defaults: defaults values for the arguments

        By specifying the group or groups arguments, only routes belonging to
        those groups will be returned.

        Routes are sorted alphabetically based on the rule.
        """
        groups_to_generate = list()
        if type(groups) is list:
            groups_to_generate = groups
        elif type(groups) is str:
            groups_to_generate.append(groups)

        links = []
        for rule in current_app.url_map.iter_rules():

            if rule.endpoint == 'static':
                continue

            func = current_app.view_functions[rule.endpoint]
            arguments = rule.arguments if rule.arguments else ['None']
            func_groups = self.func_groups[func]
            func_props = self.func_props[func] if func in self.func_props \
                else {}
            location = self.func_locations.get(func, None)

            if func_groups.intersection(groups_to_generate):
                props = dict(
                    methods=rule.methods,
                    rule="%s" % rule,
                    endpoint=rule.endpoint,
                    docstring=func.__doc__,
                    args=arguments,
                    defaults=rule.defaults,
                    location=location,
                )
                for p in func_props:
                    if p not in self.immutable_props:
                        props[p] = func_props[p]
                links.append(props)
        if sort:
            return sort(links)
        else:
            return sorted(links, key=itemgetter('rule'))