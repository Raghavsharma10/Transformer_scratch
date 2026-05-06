def stop_container(self, conf, fail_on_bad_exit=False, fail_reason=None, tag=None, remove_volumes=False):
        """Stop some container"""
        stopped = False
        container_id = conf.container_id
        if not container_id:
            return

        container_name = conf.container_name
        stopped, exit_code = self.is_stopped(conf, container_id)

        if stopped:
            if exit_code != 0 and fail_on_bad_exit:
                if not conf.harpoon.interactive:
                    print_logs = True
                else:
                    hp.write_to(conf.harpoon.stdout, "!!!!\n")
                    hp.write_to(conf.harpoon.stdout, "Container had already exited with a non zero exit code\tcontainer_name={0}\tcontainer_id={1}\texit_code={2}\n".format(container_name, container_id, exit_code))
                    hp.write_to(conf.harpoon.stdout, "Do you want to see the logs from this container?\n")
                    conf.harpoon.stdout.flush()
                    answer = input("[y]: ")
                    print_logs = not answer or answer.lower().startswith("y")

                if print_logs:
                    hp.write_to(conf.harpoon.stdout, "=================== Logs for failed container {0} ({1})\n".format(container_id, container_name))
                    logs = conf.harpoon.docker_api.logs(container_id)
                    if isinstance(logs, six.binary_type):
                        logs = logs.decode()
                    for line in logs.split("\n"):
                        hp.write_to(conf.harpoon.stdout, "{0}\n".format(line))
                    hp.write_to(conf.harpoon.stdout, "------------------- End logs for failed container\n")
                fail_reason = fail_reason or "Failed to run container"
                raise BadImage(fail_reason, container_id=container_id, container_name=container_name)
        else:
            try:
                log.info("Killing container %s:%s", container_name, container_id)
                conf.harpoon.docker_api.kill(container_id, 9)
            except DockerAPIError:
                pass
            self.wait_till_stopped(conf, container_id, timeout=10, message="waiting for container to die\tcontainer_name={0}\tcontainer_id={1}".format(container_name, container_id))

        if tag:
            log.info("Tagging a container\tcontainer_id=%s\ttag=%s", container_id, tag)
            new_id = conf.harpoon.docker_api.commit(container_id)["Id"]
            conf["committed"] = new_id
            if tag is not True:
                the_tag = "latest" if conf.tag is NotSpecified else conf.tag
                conf.harpoon.docker_api.tag(new_id, repository=tag, tag=the_tag, force=True)

        mounts = []
        if remove_volumes:
            inspection = conf.harpoon.docker_api.inspect_container(container_id)
            if "Mounts" in inspection:
                for m in inspection["Mounts"]:
                    if "Name" in m:
                        mounts.append(m['Name'])
            else:
                log.warning("Your docker can't inspect and delete volumes :(")

        if not conf.harpoon.no_cleanup:
            log.info("Removing container %s:%s", container_name, container_id)
            for _ in until(timeout=10, action="removing container\tcontainer_name={0}\tcontainer_id={1}".format(container_name, container_id)):
                try:
                    conf.harpoon.docker_api.remove_container(container_id)
                    break
                except socket.timeout:
                    break
                except (ValueError, DockerAPIError) as error:
                    log.warning("Failed to remove container\tcontainer_id=%s\terror=%s", container_id, error)

        for mount in mounts:
            try:
                log.info("Cleaning up volume {0}".format(mount))
                conf.harpoon.docker_api.remove_volume(mount)
            except DockerAPIError as error:
                log.warning("Failed to cleanup volume\tvolume=%s\terror=%s", mount, error)

        conf.harpoon.network_manager.removed(container_name)