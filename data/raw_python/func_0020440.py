def adjust_for_triggers(self):
        """Remove trigger-related plugins when needed

        If there are no triggers defined, it's assumed the
        feature is disabled and all trigger-related plugins
        are removed.

        If there are triggers defined, and this is a custom
        base image, some trigger-related plugins do not apply.

        Additionally, this method ensures that custom base
        images never have triggers since triggering a base
        image rebuild is not a valid scenario.
        """
        triggers = self.template['spec'].get('triggers', [])

        remove_plugins = [
            ("prebuild_plugins", "check_and_set_rebuild"),
            ("prebuild_plugins", "stop_autorebuild_if_disabled"),
        ]

        should_remove = False
        if triggers and (self.is_custom_base_image() or self.is_from_scratch_image()):
            if self.is_custom_base_image():
                msg = "removing %s from request because custom base image"
            elif self.is_from_scratch_image():
                msg = 'removing %s from request because FROM scratch image'
            del self.template['spec']['triggers']
            should_remove = True

        elif not triggers:
            msg = "removing %s from request because there are no triggers"
            should_remove = True

        if should_remove:
            for when, which in remove_plugins:
                logger.info(msg, which)
                self.dj.remove_plugin(when, which)