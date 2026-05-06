def deploy(self, job_name, command='', blocksize=1):
        instances = []
        """Deploy the template to a resource group."""
        self.client.resource_groups.create_or_update(
            self.resource_group,
            {
                'location': self.location,

            }
        )

        template_path = os.path.join(os.path.dirname(
            __file__), 'templates', 'template.json')
        with open(template_path, 'r') as template_file_fd:
            template = json.load(template_file_fd)

        parameters = {
            'sshKeyData': self.pub_ssh_key,
            'vmName': 'azure-deployment-sample-vm',
            'dnsLabelPrefix': self.dns_label_prefix
        }
        parameters = {k: {'value': v} for k, v in parameters.items()}

        deployment_properties = {
            'mode': DeploymentMode.incremental,
            'template': template,
            'parameters': parameters
        }
        for i in range(blocksize):
            deployment_async_operation = self.client.deployments.create_or_update(
                self.resource_group,
                'azure-sample',
                deployment_properties
            )
            instances.append(deployment_async_operation.wait())
        return instances