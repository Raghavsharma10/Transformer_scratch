def effective_task_id(self):
        """ Replace date in task id with closest date. """
        params = self.param_kwargs
        if 'date' in params and is_closest_date_parameter(self, 'date'):
            params['date'] = self.closest()
            task_id_parts = sorted(['%s=%s' % (k, str(v)) for k, v in params.items()])
            return '%s(%s)' % (self.task_family, ', '.join(task_id_parts))
        else:
            return self.task_id