def format(self, format):
    """Formats the current job into a nicer string to fit into a table."""

    job_id = "%d - %d" % (self.job.id, self.id)
    queue = self.job.queue_name if self.machine_name is None else self.machine_name
    status = "%s" % self.status + (" (%d)" % self.result if self.result is not None else "" )

    return format.format("", job_id, queue, status)