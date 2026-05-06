def delete_component(self, instance_name):
        '''Delete a component.

        Deletes the component specified by @ref instance_name from the manager.
        This will invalidate any objects that are children of this node.

        @param instance_name The instance name of the component to delete.
        @raises FailedToDeleteComponentError

        '''
        with self._mutex:
            if self._obj.delete_component(instance_name) != RTC.RTC_OK:
                raise exceptions.FailedToDeleteComponentError(instance_name)
            # The list of child components will have changed now, so it must be
            # reparsed.
            self._parse_component_children()