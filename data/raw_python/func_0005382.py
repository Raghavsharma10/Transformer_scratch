def get_filters_values(self):
        """Get different filters values as dicts."""
        # DATASETS --
        # badges
        self._DST_BADGES = requests.get(self.base_url + "datasets/badges/").json()
        # licences
        self._DST_LICENSES = {
            l.get("id"): l.get("title")
            for l in requests.get(self.base_url + "datasets/licenses").json()
        }
        # frequencies
        self._DST_FREQUENCIES = {
            f.get("id"): f.get("label")
            for f in requests.get(self.base_url + "datasets/frequencies").json()
        }
        # ORGANIZATIONS --
        # badges
        self._ORG_BADGES = requests.get(self.base_url + "organizations/badges/").json()
        # # licences
        # self._DST_LICENSES = {l.get("id"): l.get("title")
        #                   for l in requests.get(self.base_url + "datasets/licenses").json()}
        # # frequencies
        # self._DST_FREQUENCIES = {f.get("id"): f.get("label")
        #                      for f in requests.get(self.base_url + "datasets/frequencies").json()}
        # SPATIAL --
        # granularities
        self._GRANULARITIES = {
            g.get("id"): g.get("name")
            for g in requests.get(self.base_url + "spatial/granularities").json()
        }
        # levels
        self._LEVELS = {
            g.get("id"): g.get("name")
            for g in requests.get(self.base_url + "spatial/levels").json()
        }
        # MISC --
        # facets
        self._FACETS = (
            "all",
            "badge",
            "featured",
            "format",
            "geozone",
            "granularity",
            "license",
            "owner",
            "organization",
            "reuses",
            "tag",
            "temporal_coverage",
        )
        # reuses
        self._REUSES = ("none", "few", "quite", "many")