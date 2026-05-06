def angularFrontendAppDir(self) -> str:
        """ Angular Frontend Dir

        This directory will be linked into the angular app when it is compiled.

        :return: The absolute path of the Angular2 app directory.
        """
        relDir = self._packageCfg.config.plugin.title(require_string)
        dir = os.path.join(self._pluginRoot, relDir)
        if not os.path.isdir(dir): raise NotADirectoryError(dir)
        return dir