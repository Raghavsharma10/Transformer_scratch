def set_client_cmds(self):
        """
        This is method automatically called on each request and
        updates "object_id", "cmd" and "flow"  client variables
        from current.input.

        "flow" and "object_id" variables will always exists in the
        task_data so app developers can safely check for their
        values in workflows.
        Their values will be reset to None if they not exists
        in the current input data set.

        On the other side, if there isn't a "cmd" in the current.input
        cmd will be removed from task_data.

        """
        self.task_data['cmd'] = self.input.get('cmd')

        self.task_data['flow'] = self.input.get('flow')

        filters = self.input.get('filters', {})

        try:
            if isinstance(filters, dict):
                # this is the new form, others will be removed when ui be ready
                self.task_data['object_id'] = filters.get('object_id')['values'][0]
            elif filters[0]['field'] == 'object_id':
                self.task_data['object_id'] = filters[0]['values'][0]
        except:
            if 'object_id' in self.input:
                self.task_data['object_id'] = self.input.get('object_id')