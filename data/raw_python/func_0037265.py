async def get_pypi_version(self):
        """Get version published to PyPi."""
        self._version_data["beta"] = self.beta
        self._version_data["source"] = "PyPi"

        info_version = None
        last_release = None

        try:
            async with async_timeout.timeout(5, loop=self.loop):
                response = await self.session.get(URL["pypi"])
            data = await response.json()

            info_version = data["info"]["version"]
            releases = data["releases"]

            for version in sorted(releases, reverse=True):
                if re.search(r"^(\\d+\\.)?(\\d\\.)?(\\*|\\d+)$", version):
                    continue
                else:
                    last_release = version
                    break

            self._version = info_version

            if self.beta:
                if info_version in last_release:
                    self._version = info_version
                else:
                    self._version = last_release

            _LOGGER.debug("Version: %s", self.version)
            _LOGGER.debug("Version data: %s", self.version_data)
        except asyncio.TimeoutError as error:
            _LOGGER.error("Timeouterror fetching version information from PyPi")
        except KeyError as error:
            _LOGGER.error("Error parsing version information from PyPi, %s", error)
        except TypeError as error:
            _LOGGER.error("Error parsing version information from PyPi, %s", error)
        except aiohttp.ClientError as error:
            _LOGGER.error("Error fetching version information from PyPi, %s", error)
        except socket.gaierror as error:
            _LOGGER.error("Error fetching version information from PyPi, %s", error)
        except Exception as error:  # pylint: disable=broad-except
            _LOGGER.critical("Something really wrong happend! - %s", error)