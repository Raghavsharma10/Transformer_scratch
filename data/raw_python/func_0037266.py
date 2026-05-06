async def get_hassio_version(self):
        """Get version published for hassio."""
        if self.image not in IMAGES:
            _LOGGER.warning("%s is not a valid image using default", self.image)
            self.image = "default"

        board = BOARDS.get(self.image, BOARDS["default"])

        self._version_data["source"] = "Hassio"
        self._version_data["beta"] = self.beta
        self._version_data["board"] = board
        self._version_data["image"] = IMAGES[self.image]["hassio"]

        try:
            async with async_timeout.timeout(5, loop=self.loop):
                response = await self.session.get(
                    URL["hassio"]["beta" if self.beta else "stable"]
                )
                data = await response.json()

                self._version = data["homeassistant"][IMAGES[self.image]["hassio"]]

                self._version_data["hassos"] = data["hassos"][board]
                self._version_data["supervisor"] = data["supervisor"]
                self._version_data["hassos-cli"] = data["hassos-cli"]

            _LOGGER.debug("Version: %s", self.version)
            _LOGGER.debug("Version data: %s", self.version_data)
        except asyncio.TimeoutError as error:
            _LOGGER.error("Timeouterror fetching version information for hassio")
        except KeyError as error:
            _LOGGER.error("Error parsing version information for hassio, %s", error)
        except TypeError as error:
            _LOGGER.error("Error parsing version information for hassio, %s", error)
        except aiohttp.ClientError as error:
            _LOGGER.error("Error fetching version information for hassio, %s", error)
        except socket.gaierror as error:
            _LOGGER.error("Error fetching version information for hassio, %s", error)
        except Exception as error:  # pylint: disable=broad-except
            _LOGGER.critical("Something really wrong happend! - %s", error)