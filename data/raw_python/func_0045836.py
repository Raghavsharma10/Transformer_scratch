def wait_till_stopped(self, conf, container_id, timeout=10, message=None, waiting=True):
        """Wait till a container is stopped"""
        stopped = False
        inspection = None
        for _ in until(timeout=timeout, action=message):
            try:
                inspection = conf.harpoon.docker_api.inspect_container(container_id)
                if not isinstance(inspection, dict):
                    log.error("Weird response from inspecting the container\tresponse=%s", inspection)
                else:
                    if not inspection["State"]["Running"]:
                        stopped = True
                        conf.container_id = None
                        break
                    else:
                        break
            except (socket.timeout, ValueError):
                log.warning("Failed to inspect the container\tcontainer_id=%s", container_id)
            except DockerAPIError as error:
                if error.response.status_code != 404:
                    raise
                else:
                    break

        if not inspection:
            log.warning("Failed to inspect the container!")
            stopped = True
            exit_code = 1
        else:
            exit_code = inspection["State"]["ExitCode"]
        return stopped, exit_code