def get_tasks_changed_since(self, since):
        """ Returns a list of tasks that were changed recently."""
        changed_tasks = []

        for task in self.client.filter_tasks({'status': 'pending'}):
            if task.get(
                'modified',
                task.get(
                    'entry',
                    datetime.datetime(2000, 1, 1).replace(tzinfo=pytz.utc)
                )
            ) >= since:
                changed_tasks.append(task)

        return changed_tasks