def create_component(self, module_name):
        '''Create a component out of a loaded module.

        Turns a previously-loaded shared module into a component in the
        manager. This will invalidate any objects that are children of this
        node.

        The @ref module_name argument can contain options that set various
        properties of the new component. These must be appended to the module
        name, prefixed by a question mark for each property, in key=value
        format. For example, to change the instance name of the new component,
        append '?instance_name=new_name' to the module name.

        @param module_name Name of the module to turn into a component.
        @raises FailedToCreateComponentError

        '''
        with self._mutex:
            if not self._obj.create_component(module_name):
                raise exceptions.FailedToCreateComponentError(module_name)
            # The list of child components will have changed now, so it must be
            # reparsed.
            self._parse_component_children()