def get_exit_code(self, conf):
        """Determine how a container exited"""
        for _ in until(timeout=0.5, step=0.1, silent=True):
            try:
                inspection = conf.harpoon.docker_api.inspect_container(conf.container_id)
                if not isinstance(inspection, dict) or "State" not in inspection:
                    raise BadResult("Expected inspect result to be a dictionary with 'State' in it", found=inspection)
                elif not inspection["State"]["Running"]:
                    return inspection
            except Exception as error:
                log.error("Failed to see if container exited normally or not\thash=%s\terror=%s", conf.container_id, error)