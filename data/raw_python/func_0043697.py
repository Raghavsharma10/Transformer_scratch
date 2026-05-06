def destroy(self, job_ids):
        """Destroy the given resource group"""
        for job_id in job_ids:
            self.client.resource_groups.delete(self.resource_group)