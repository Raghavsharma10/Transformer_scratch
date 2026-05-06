def run_watcher(self):
        """
        Watcher thread's function.

        :return:
            None.
        """
        # Create observer
        observer = Observer()

        # Start observer
        observer.start()

        # Dict that maps file path to `watch object`
        watche_obj_map = {}

        # Run change check in a loop
        while not self._watcher_to_stop:
            # Get current watch paths
            old_watch_path_s = set(watche_obj_map)

            # Get new watch paths
            new_watch_path_s = self._find_watch_paths()

            # For each new watch path
            for new_watch_path in new_watch_path_s:
                # Remove from the old watch paths if exists
                old_watch_path_s.discard(new_watch_path)

                # If the new watch path was not watched
                if new_watch_path not in watche_obj_map:
                    try:
                        # Schedule a watch
                        watch_obj = observer.schedule(
                            # 2KGRW
                            # `FileSystemEventHandler` instance
                            self,
                            # File path to watch
                            new_watch_path,
                            # Whether recursive
                            recursive=True,
                        )

                        # Store the watch obj
                        watche_obj_map[new_watch_path] = watch_obj

                    # If have error
                    except OSError:
                        # Set the watch object be None
                        watche_obj_map[new_watch_path] = None

            # For each old watch path that is not in the new watch paths
            for old_watch_path in old_watch_path_s:
                # Get watch object
                watch_obj = watche_obj_map.pop(old_watch_path, None)

                # If have watch object
                if watch_obj is not None:
                    # Unschedule the watch
                    observer.unschedule(watch_obj)

            # Store new watch paths
            self._watch_paths = new_watch_path_s

            # Sleep before next check
            time.sleep(self._interval)