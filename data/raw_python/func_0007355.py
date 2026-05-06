def up_to_date(self):
        """Check if Team Password Manager is up to date."""
        VersionInfo = self.get_latest_version()
        CurrentVersion = VersionInfo.get('version')
        LatestVersion = VersionInfo.get('latest_version')
        if  CurrentVersion == LatestVersion:
            log.info('TeamPasswordManager is up-to-date!')
            log.debug('Current Version: {} Latest Version: {}'.format(LatestVersion, LatestVersion))
            return True
        else:
            log.warning('TeamPasswordManager is not up-to-date!')
            log.debug('Current Version: {} Latest Version: {}'.format(LatestVersion, LatestVersion))
            return False