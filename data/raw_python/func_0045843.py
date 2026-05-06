def commit_and_run(self, commit, conf, command="sh"):
        """Commit this container id and run the provided command in it and clean up afterwards"""
        image_hash = None
        try:
            image_hash = conf.harpoon.docker_api.commit(commit)["Id"]

            new_conf = conf.clone()
            new_conf.bash = NotSpecified
            new_conf.command = command
            new_conf.image_name = image_hash
            new_conf.container_id = None
            new_conf.container_name = "{0}-intervention-{1}".format(conf.container_id, str(uuid.uuid1()))

            container_id = self.create_container(new_conf, False, True)
            new_conf.container_id = container_id

            try:
                self.start_container(new_conf, tty=True, detach=False, is_dependency=False, no_intervention=True)
            finally:
                self.stop_container(new_conf)
            yield
        except Exception as error:
            log.error("Something failed about creating the intervention image\terror=%s", error)
            raise
        finally:
            try:
                if image_hash:
                    log.info("Removing intervened image\thash=%s", image_hash)
                    conf.harpoon.docker_api.remove_image(image_hash)
            except Exception as error:
                log.error("Failed to kill intervened image\thash=%s\terror=%s", image_hash, error)