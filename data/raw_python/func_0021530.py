def send(self, message, channel_name=None, fail_silently=False,
             options=None):
        # type: (Text, Optional[str], bool, Optional[SendOptions]) -> None
        """Send a notification to channels

        :param message: A message
        """
        if channel_name is None:
            channels = self.settings["CHANNELS"]
        else:
            try:
                channels = {
                    "__selected__": self.settings["CHANNELS"][channel_name]
                }
            except KeyError:
                raise Exception("channels does not exist %s", channel_name)

        for _, config in channels.items():
            if "_backend" not in config:
                raise ImproperlyConfigured(
                    "Specify the backend class in the channel configuration")

            backend = self._load_backend(config["_backend"])  # type: Any
            config = deepcopy(config)
            del config["_backend"]
            channel = backend(**config)
            channel.send(message, fail_silently=fail_silently, options=options)