def _taint_deployment(self, stream=False):
        """
        Run 'terraform taint aws_api_gateway_deployment.depl' to taint the
        deployment resource. This is a workaround for
        https://github.com/hashicorp/terraform/issues/6613

        :param stream: whether or not to stream TF output in realtime
        :type stream: bool
        """
        args = ['aws_api_gateway_deployment.depl']
        logger.warning('Running terraform taint: %s as workaround for '
                       '<https://github.com/hashicorp/terraform/issues/6613>',
                       ' '.join(args))
        out = self._run_tf('taint', cmd_args=args, stream=stream)
        if stream:
            logger.warning('Terraform taint finished successfully.')
        else:
            logger.warning("Terraform taint finished successfully:\n%s", out)