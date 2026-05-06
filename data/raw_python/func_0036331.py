def start(self, on_add, on_update, on_delete):
        """
        Starts monitoring the file path, passing along on_(add|update|delete)
        callbacks to a watchdog observer.

        Iterates over the files in the target path before starting the observer
        and calls the on_created callback before starting the observer, so
        that existing files aren't missed.
        """
        handler = ConfigFileChangeHandler(
            self.target_class, on_add, on_update, on_delete
        )

        for file_name in os.listdir(self.file_path):
            if os.path.isdir(os.path.join(self.file_path, file_name)):
                continue
            if (
                not self.target_class.config_subdirectory and
                not (
                    file_name.endswith(".yaml") or file_name.endswith(".yml")
                )
            ):
                continue

            handler.on_created(
                events.FileCreatedEvent(
                    os.path.join(self.file_path, file_name)
                )
            )

        observer = observers.Observer()
        observer.schedule(handler, self.file_path)
        observer.start()

        return observer