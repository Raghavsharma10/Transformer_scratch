def doc(self, groups=None, set_location=True, **properties):
        """Add flask route to autodoc for automatic documentation

        Any route decorated with this method will be added to the list of
        routes to be documented by the generate() or html() methods.

        By default, the route is added to the 'all' group.
        By specifying group or groups argument, the route can be added to one
        or multiple other groups as well, besides the 'all' group.

        If set_location is True, the location of the function will be stored.
        NOTE: this assumes that the decorator is placed just before the
        function (in the normal way).

        Custom parameters may also be passed in beyond groups, if they are
        named something not already in the dict descibed in the docstring for
        the generare() function, they will be added to the route's properties,
        which can be accessed from the template.

        If a parameter is passed in with a name that is already in the dict, but
        not of a reserved name, the passed parameter overrides that dict value.
        """
        def decorator(f):
            # Get previous group list (if any)
            if f in self.func_groups:
                groupset = self.func_groups[f]
            else:
                groupset = set()

            # Set group[s]
            if type(groups) is list:
                groupset.update(groups)
            elif type(groups) is str:
                groupset.add(groups)
            groupset.add('all')
            self.func_groups[f] = groupset
            self.func_props[f] = properties

            # Set location
            if set_location:
                caller_frame = inspect.stack()[1]
                self.func_locations[f] = {
                        'filename': caller_frame[1],
                        'line':     caller_frame[2],
                        }

            return f
        return decorator