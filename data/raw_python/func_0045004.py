def load_plugin(self, manifest, *args):
        """
        Loads a plugin from the given manifest

        :param manifest: The manifest to use to load the plugin
        :param args: Arguments to pass to the plugin
        """
        if self.get_plugin_loaded(manifest["name"]):
            self._logger.debug("Plugin {} is already loaded.".format(manifest["name"]))
            return
        try:
            self._logger.debug("Attempting to load plugin {}.".format(manifest["name"]))
            for dependency in manifest.get("dependencies", []):
                if not self.get_plugin_loaded(dependency):
                    self._logger.debug("Must load dependency {} first.".format(dependency))
                    if self.get_manifest(dependency) is None:
                        self._logger.error("Dependency {} could not be found.".format(dependency))
                    else:
                        self.load_plugin(self.get_manifest(dependency), *args)

            not_loaded = [i for i in manifest.get("dependencies", []) if not self.get_plugin_loaded(i)]
            if len(not_loaded) != 0:
                self._logger.error("Plugin {} failed to load due to missing dependencies. Dependencies: {}".format(
                    manifest["name"], ", ".join(not_loaded)
                ))
                return

            if PY3:
                spec = importlib.util.spec_from_file_location(
                    manifest.get("module_name", manifest["name"].replace(" ", "_")),
                    os.path.join(manifest["path"], manifest.get("main_path", "__init__.py"))
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                module = imp.load_source(
                    manifest.get("module_name", manifest["name"].replace(" ", "_")),
                    os.path.join(manifest["path"], manifest.get("main_path", "__init__.py"))
                )

            module_class = manifest.get("main_class", "Plugin")
            plugin_class = getattr(module, module_class)
            if issubclass(plugin_class, self._plugin_class):
                plugin = plugin_class(manifest, *args)
            else:
                self._logger.error("Failed to load {} due to invalid baseclass.".format(manifest["name"]))
                return
            self._plugins[manifest["name"]] = plugin
            self._modules[manifest["name"]] = module

            self._logger.debug("Plugin {} loaded.".format(manifest["name"]))

        except:
            exc_path = os.path.join(manifest["path"], "error.log")
            with open(exc_path, "w") as f:
                f.write(traceback.format_exc(5))
            self._logger.error("Failed to load plugin {}. Error log written to {}.".format(manifest["name"], exc_path))