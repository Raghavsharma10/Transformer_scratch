def dependency_images(self, for_running=False):
        """
        What images does this one require

        Taking into account parent image, and those in link and volumes.share_with options
        """
        candidates = []
        detach = dict((candidate, not options.attached) for candidate, options in self.dependency_options.items())

        for link in self.links:
            if link.container:
                candidates.append(link.container.name)

        if not for_running:
            for content, _ in self.commands.extra_context:
                if type(content) is dict or (hasattr(content, "is_dict") and content.is_dict) and "image" in content:
                    if not isinstance(content["image"], six.string_types):
                        candidates.append(content["image"].name)

        candidates.extend(list(self.shared_volume_containers()))

        done = []
        for candidate in candidates:
            if candidate not in done:
                done.append(candidate)
                yield candidate, detach.get(candidate, True)