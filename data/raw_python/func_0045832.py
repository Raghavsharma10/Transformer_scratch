def wait_for_dep(self, api, conf, wait_condition, start, last_attempt):
        """Wait for this image"""
        from harpoon.option_spec.image_objs import WaitCondition
        conditions = list(wait_condition.conditions(start, last_attempt))
        if conditions[0] in (WaitCondition.KeepWaiting, WaitCondition.Timedout):
            return conditions[0]

        log.info("Waiting for %s", conf.container_name)
        for condition in conditions:
            log.debug("Running condition\tcondition=%s", condition)
            command = 'bash -c "{0}"'.format(condition)
            try:
                exec_id = api.exec_create(conf.container_id, command, tty=False)
            except DockerAPIError as error:
                log.error("Failed to run condition\tcondition=%s\tdependency=%s\terror=%s", condition, conf.name, error)
                return False

            output = api.exec_start(exec_id).decode('utf-8')
            inspection = api.exec_inspect(exec_id)
            exit_code = inspection["ExitCode"]
            if exit_code != 0:
                log.error("Condition says no\tcondition=%s\toutput:\n\t%s", condition, "\n\t".join(line for line in output.split('\n')))
                return False

        log.info("Finished waiting for %s", conf.container_name)
        return True