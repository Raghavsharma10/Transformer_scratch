async def get_docker_version(self):
        """Get version published for docker."""
        if self.image not in IMAGES:
            _LOGGER.warning("%s is not a valid image using default", self.image)
            self.image = "default"

        self._version_data["beta"] = self.beta
        self._version_data["source"] = "Docker"
        self._version_data["image"] = IMAGES[self.image]["docker"]
        try:
            async with async_timeout.timeout(5, loop=self.loop):
                response = await self.session.get(
                    URL["docker"].format(IMAGES[self.image]["docker"])
                )
                data = await response.json()
                for version in data["results"]:
                    if version["name"] in ["latest", "landingpage", "rc", "dev"]:
                        continue
                    elif re.search(r"\b.+b\d", version["name"]):
                        if self.beta:
                            self._version = version["name"]
                            break
                        else:
                            continue
                    else:
                        self._version = version["name"]

                    if self._version is not None:
                        break
                    else:
                        continue

            _LOGGER.debug("Version: %s", self.version)
            _LOGGER.debug("Version data: %s", self.version_data)
        except asyncio.TimeoutError as error:
            _LOGGER.error("Timeouterror fetching version information for docker")
        except KeyError as error:
            _LOGGER.error("Error parsing version information for docker, %s", error)
        except TypeError as error:
            _LOGGER.error("Error parsing version information for docker, %s", error)
        except aiohttp.ClientError as error:
            _LOGGER.error("Error fetching version information for docker, %s", error)
        except socket.gaierror as error:
            _LOGGER.error("Error fetching version information for docker, %s", error)
        except Exception as error:  # pylint: disable=broad-except
            _LOGGER.critical("Something really wrong happend! - %s", error)