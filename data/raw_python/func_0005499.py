def set_deployment_home(self):
        """Sets self.deployment_home

        This method finds and sets deployment home, primarily based on
        the DEPLOYMENT_HOME environment variable. If not set, this
        method will attempt to determine deployment home.

        :return: None
        """
        log = logging.getLogger(self.cls_logger + '.set_deployment_home')
        try:
            self.deployment_home = os.environ['DEPLOYMENT_HOME']
        except KeyError:
            log.warn('DEPLOYMENT_HOME environment variable is not set, attempting to set it...')
        else:
            log.info('Found DEPLOYMENT_HOME environment variable set to: {d}'.format(d=self.deployment_home))
            return

        if self.cons3rt_agent_run_dir is None:
            msg = 'This is not Windows nor Linux, cannot determine DEPLOYMENT_HOME'
            log.error(msg)
            raise DeploymentError(msg)

        # Ensure the run directory can be found
        if not os.path.isdir(self.cons3rt_agent_run_dir):
            msg = 'Could not find the cons3rt run directory, DEPLOYMENT_HOME cannot be set'
            log.error(msg)
            raise DeploymentError(msg)

        run_dir_contents = os.listdir(self.cons3rt_agent_run_dir)
        results = []
        for item in run_dir_contents:
            if 'Deployment' in item:
                results.append(item)
        if len(results) != 1:
            msg = 'Could not find deployment home in the cons3rt run directory, deployment home cannot be set'
            log.error(msg)
            raise DeploymentError(msg)

        # Ensure the Deployment Home is a directory
        candidate_deployment_home = os.path.join(self.cons3rt_agent_run_dir, results[0])
        if not os.path.isdir(candidate_deployment_home):
            msg = 'The candidate deployment home is not a valid directory: {d}'.format(d=candidate_deployment_home)
            log.error(msg)
            raise DeploymentError(msg)

        # Ensure the deployment properties file can be found
        self.deployment_home = candidate_deployment_home
        os.environ['DEPLOYMENT_HOME'] = self.deployment_home
        log.info('Set DEPLOYMENT_HOME in the environment to: {d}'.format(d=self.deployment_home))