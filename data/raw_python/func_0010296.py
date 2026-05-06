def send_sci_async(self, operation, target, payload, **sci_options):
        """Send an asynchronous SCI request, and wraps the job in an object
        to manage it

        :param str operation: The operation is one of {send_message, update_firmware, disconnect, query_firmware_targets,
            file_system, data_service, and reboot}
        :param target: The device(s) to be targeted with this request
        :type target: :class:`~.TargetABC` or list of :class:`~.TargetABC` instances

        TODO: document other params

        """
        sci_options['synchronous'] = False
        resp = self.send_sci(operation, target, payload, **sci_options)
        dom = ET.fromstring(resp.content)
        job_element = dom.find('.//jobId')
        if job_element is None:
            return
        job_id = int(job_element.text)
        return AsyncRequestProxy(job_id, self._conn)