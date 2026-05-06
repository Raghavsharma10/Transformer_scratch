def run_edit_script(self):
        """
        Run an xml edit script, and return the new html produced.
        """
        for action, location, properties in self.edit_script:
            if action == 'delete':
                node = get_location(self.dom, location)
                self.action_delete(node)
            elif action == 'insert':
                parent = get_location(self.dom, location[:-1])
                child_index = location[-1]
                self.action_insert(parent, child_index, **properties)
        return self.dom