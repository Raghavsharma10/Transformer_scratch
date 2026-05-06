def get_download_urls(self, package_name, version="", pkg_type="all"):
        """Query PyPI for pkg download URI for a packge"""

        if version:
            versions = [version]
        else:

            #If they don't specify version, show em all.

            (package_name, versions) = self.query_versions_pypi(package_name)

        all_urls = []
        for ver in versions:
            metadata = self.release_data(package_name, ver)
            for urls in self.release_urls(package_name, ver):
                if pkg_type == "source" and urls['packagetype'] == "sdist":
                    all_urls.append(urls['url'])
                elif pkg_type == "egg" and \
                        urls['packagetype'].startswith("bdist"):
                    all_urls.append(urls['url'])
                elif pkg_type == "all":
                    #All
                    all_urls.append(urls['url'])

            #Try the package's metadata directly in case there's nothing
            #returned by XML-RPC's release_urls()
            if metadata and metadata.has_key('download_url') and \
                        metadata['download_url'] != "UNKNOWN" and \
                        metadata['download_url'] != None:
                if metadata['download_url'] not in all_urls:
                    if pkg_type != "all":
                        url = filter_url(pkg_type, metadata['download_url'])
                        if url:
                            all_urls.append(url)
        return all_urls