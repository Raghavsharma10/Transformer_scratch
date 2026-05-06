def to_dict(self):
        """ Return the task context content as a dictionary. """
        return {
            'task_name': self.task_name,
            'dag_name': self.dag_name,
            'workflow_name': self.workflow_name,
            'workflow_id': self.workflow_id,
            'worker_hostname': self.worker_hostname
        }