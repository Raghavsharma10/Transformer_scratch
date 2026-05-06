def wait_for_deps(self, conf, images):
        """Wait for all our dependencies"""
        from harpoon.option_spec.image_objs import WaitCondition
        api = conf.harpoon.docker_context_maker().api

        waited = set()
        last_attempt = {}
        dependencies = set(dep for dep, _ in conf.dependency_images())

        # Wait conditions come from dependency_options first
        # Or if none specified there, they come from the image itself
        wait_conditions = {}
        for dependency in dependencies:
            if conf.dependency_options is not NotSpecified and dependency in conf.dependency_options and conf.dependency_options[dependency].wait_condition is not NotSpecified:
                wait_conditions[dependency] = conf.dependency_options[dependency].wait_condition
            elif images[dependency].wait_condition is not NotSpecified:
                wait_conditions[dependency] = images[dependency].wait_condition

        if not wait_conditions:
            return

        start = time.time()
        while True:
            this_round = []
            for dependency in dependencies:
                if dependency in waited:
                    continue

                image = images[dependency]
                if dependency in wait_conditions:
                    done = self.wait_for_dep(api, image, wait_conditions[dependency], start, last_attempt.get(dependency))
                    this_round.append(done)
                    if done is True:
                        waited.add(dependency)
                    elif done is False:
                        last_attempt[dependency] = time.time()
                    elif done is WaitCondition.Timedout:
                        log.warning("Stopping dependency because it timedout waiting\tcontainer_id=%s", image.container_id)
                        self.stop_container(image)
                else:
                    waited.add(dependency)

            if set(this_round) != set([WaitCondition.KeepWaiting]):
                if dependencies - waited == set():
                    log.info("Finished waiting for dependencies")
                    break
                else:
                    log.info("Still waiting for dependencies\twaiting_on=%s", list(dependencies-waited))

                couldnt_wait = set()
                container_ids = {}
                for dependency in dependencies:
                    if dependency in waited:
                        continue

                    image = images[dependency]
                    if image.container_id is None:
                        stopped = True
                        if dependency not in container_ids:
                            available = sorted([i for i in available if "/{0}".format(image.container_name) in i["Names"]], key=lambda i: i["Created"])
                            if available:
                                container_ids[dependency] = available[0]["Id"]
                    else:
                        if dependency not in container_ids:
                            container_ids[dependency] = image.container_id
                        stopped, _ = self.is_stopped(image, image.container_id)

                    if stopped:
                        couldnt_wait.add(dependency)

                if couldnt_wait:
                    for container in couldnt_wait:
                        if container not in images or container not in container_ids:
                            continue
                        image = images[container]
                        container_id = container_ids[container]
                        container_name = image.container_name
                        hp.write_to(conf.harpoon.stdout, "=================== Logs for failed container {0} ({1})\n".format(container_id, container_name))
                        for line in conf.harpoon.docker_api.logs(container_id).split("\n"):
                            hp.write_to(conf.harpoon.stdout, "{0}\n".format(line))
                        hp.write_to(conf.harpoon.stdout, "------------------- End logs for failed container\n")
                    raise BadImage("One or more of the dependencies stopped running whilst waiting for other dependencies", stopped=list(couldnt_wait))

            time.sleep(0.1)