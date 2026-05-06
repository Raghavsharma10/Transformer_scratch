def _get_annual_data(self, p_p_id):
        """Get annual data."""
        params = {"p_p_id": p_p_id,
                  "p_p_lifecycle": 2,
                  "p_p_state": "normal",
                  "p_p_mode": "view",
                  "p_p_resource_id": "resourceObtenirDonneesConsommationAnnuelles"}
        try:
            raw_res = yield from self._session.get(PROFILE_URL,
                                                   params=params,
                                                   timeout=self._timeout)
        except OSError:
            raise PyHydroQuebecAnnualError("Can not get annual data")
        try:
            json_output = yield from raw_res.json(content_type='text/json')
        except (OSError, json.decoder.JSONDecodeError):
            raise PyHydroQuebecAnnualError("Could not get annual data")

        if not json_output.get('success'):
            raise PyHydroQuebecAnnualError("Could not get annual data")

        if not json_output.get('results'):
            raise PyHydroQuebecAnnualError("Could not get annual data")

        if 'courant' not in json_output.get('results')[0]:
            raise PyHydroQuebecAnnualError("Could not get annual data")

        return json_output.get('results')[0]['courant']