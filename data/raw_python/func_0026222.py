def _get_monthly_data(self, p_p_id):
        """Get monthly data."""
        params = {"p_p_id": p_p_id,
                  "p_p_lifecycle": 2,
                  "p_p_resource_id": ("resourceObtenirDonnees"
                                      "PeriodesConsommation")}
        try:
            raw_res = yield from self._session.get(PROFILE_URL,
                                                   params=params,
                                                   timeout=self._timeout)
        except OSError:
            raise PyHydroQuebecError("Can not get monthly data")
        try:
            json_output = yield from raw_res.json(content_type='text/json')
        except (OSError, json.decoder.JSONDecodeError):
            raise PyHydroQuebecError("Could not get monthly data")

        if not json_output.get('success'):
            raise PyHydroQuebecError("Could not get monthly data")

        return json_output.get('results')