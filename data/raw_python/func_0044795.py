def _generate_api_gateway_deployment(self):
        """
        Generate the API Gateway Deployment/Stage, and add to self.tf_conf
        """
        # finally, the deployment
        # this resource MUST come last
        dep_on = []
        for rtype in sorted(self.tf_conf['resource'].keys()):
            for rname in sorted(self.tf_conf['resource'][rtype].keys()):
                dep_on.append('%s.%s' % (rtype, rname))
        self.tf_conf['resource']['aws_api_gateway_deployment']['depl'] = {
            'rest_api_id': '${aws_api_gateway_rest_api.rest_api.id}',
            'description': self.description,
            'stage_name': self.config.stage_name,
            'depends_on': dep_on
        }
        self.tf_conf['output']['deployment_id'] = {
            'value': '${aws_api_gateway_deployment.depl.id}'
        }