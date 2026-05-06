def register(self, notification_cls=None):
        """Registers a Notification class unique by name.
        """
        self.loaded = True
        display_names = [n.display_name for n in self.registry.values()]
        if (
            notification_cls.name not in self.registry
            and notification_cls.display_name not in display_names
        ):
            self.registry.update({notification_cls.name: notification_cls})

            models = getattr(notification_cls, "models", [])
            if not models and getattr(notification_cls, "model", None):
                models = [getattr(notification_cls, "model")]
            for model in models:
                try:
                    if notification_cls.name not in [
                        n.name for n in self.models[model]
                    ]:
                        self.models[model].append(notification_cls)
                except KeyError:
                    self.models.update({model: [notification_cls]})
        else:
            raise AlreadyRegistered(
                f"Notification {notification_cls.name}: "
                f"{notification_cls.display_name} is already registered."
            )