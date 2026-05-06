def from_celery(cls, worker_name, job_dict, celery_app):
        """ Create a JobStats object from the dictionary returned by celery.

        Args:
            worker_name (str): The name of the worker this jobs runs on.
            job_dict (dict): The dictionary as returned by celery.
            celery_app: Reference to a celery application object.

        Returns:
            JobStats: A fully initialized JobStats object.
        """
        if not isinstance(job_dict, dict) or 'id' not in job_dict:
            raise JobStatInvalid('The job description is missing important fields.')

        async_result = AsyncResult(id=job_dict['id'], app=celery_app)
        a_info = async_result.info if isinstance(async_result.info, dict) else None

        return JobStats(
            name=a_info.get('name', '') if a_info is not None else '',
            job_id=job_dict['id'],
            job_type=a_info.get('type', '') if a_info is not None else '',
            workflow_id=a_info.get('workflow_id', '') if a_info is not None else '',
            queue=a_info.get('queue', '') if a_info is not None else '',
            start_time=a_info.get('start_time', None) if a_info is not None else None,
            arguments=a_info.get('arguments', {}) if a_info is not None else {},
            acknowledged=job_dict['acknowledged'],
            func_name=job_dict['type'],
            hostname=job_dict['hostname'],
            worker_name=worker_name,
            worker_pid=job_dict['worker_pid'],
            routing_key=job_dict['delivery_info']['routing_key']
        )