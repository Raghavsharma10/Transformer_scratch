async def get_local_version(self):
        """Get the local installed version."""
        self._version_data["source"] = "Local"
        try:
            from homeassistant.const import __version__ as localversion

            self._version = localversion

            _LOGGER.debug("Version: %s", self.version)
            _LOGGER.debug("Version data: %s", self.version_data)
        except ImportError as error:
            _LOGGER.critical("Home Assistant not found - %s", error)
        except Exception as error:  # pylint: disable=broad-except
            _LOGGER.critical("Something really wrong happend! - %s", error)