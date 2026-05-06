def wait(self, build_id, states):
        """
        :param build_id: wait for build to finish

        :return:
        """
        logger.info("watching build '%s'", build_id)
        for changetype, obj in self.watch_resource("builds", build_id):
            try:
                obj_name = obj["metadata"]["name"]
            except KeyError:
                logger.error("'object' doesn't have any name")
                continue
            try:
                obj_status = obj["status"]["phase"]
            except KeyError:
                logger.error("'object' doesn't have any status")
                continue
            else:
                obj_status_lower = obj_status.lower()
            logger.info("object has changed: '%s', status: '%s', name: '%s'",
                        changetype, obj_status, obj_name)
            if obj_name == build_id:
                logger.info("matching build found")
                logger.debug("is %s in %s?", repr(obj_status_lower), states)
                if obj_status_lower in states:
                    logger.debug("Yes, build is in the state I'm waiting for.")
                    return obj
                else:
                    logger.debug("No, build is not in the state I'm "
                                 "waiting for.")
            else:
                logger.info("The build %r isn't me %r", obj_name, build_id)

        # I'm not sure how we can end up here since there are two possible scenarios:
        #   1. our object was found and we are returning in the loop
        #   2. our object was not found and we keep waiting (in the loop)
        # Therefore, let's raise here
        logger.warning("build '%s' was not found during wait", build_id)
        raise OsbsWatchBuildNotFound("build '%s' was not found and response stream ended" %
                                     build_id)