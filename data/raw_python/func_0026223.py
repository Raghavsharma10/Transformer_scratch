def _get_hourly_data(self, day_date, p_p_id):
        """Get Hourly Data."""
        params = {"p_p_id": p_p_id,
                  "p_p_lifecycle": 2,
                  "p_p_state": "normal",
                  "p_p_mode": "view",
                  "p_p_resource_id": "resourceObtenirDonneesConsommationHoraires",
                  "p_p_cacheability": "cacheLevelPage",
                  "p_p_col_id": "column-2",
                  "p_p_col_count": 1,
                  "date": day_date,
                  }
        try:
            raw_res = yield from self._session.get(PROFILE_URL,
                                                   params=params,
                                                   timeout=self._timeout)
        except OSError:
            raise PyHydroQuebecError("Can not get hourly data")
        try:
            json_output = yield from raw_res.json(content_type='text/json')
        except (OSError, json.decoder.JSONDecodeError):
            raise PyHydroQuebecAnnualError("Could not get hourly data")
        hourly_consumption_data = json_output['results']['listeDonneesConsoEnergieHoraire']
        hourly_power_data = json_output['results']['listeDonneesConsoPuissanceHoraire']
        params = {"p_p_id": p_p_id,
                  "p_p_lifecycle": 2,
                  "p_p_state": "normal",
                  "p_p_mode": "view",
                  "p_p_resource_id": "resourceObtenirDonneesMeteoHoraires",
                  "p_p_cacheability": "cacheLevelPage",
                  "p_p_col_id": "column-2",
                  "p_p_col_count": 1,
                  "dateDebut": day_date,
                  "dateFin": day_date,
                  }
        try:
            raw_res = yield from self._session.get(PROFILE_URL,
                                                   params=params,
                                                   timeout=self._timeout)
        except OSError:
            raise PyHydroQuebecError("Can not get hourly data")
        try:
            json_output = yield from raw_res.json(content_type='text/json')
        except (OSError, json.decoder.JSONDecodeError):
            raise PyHydroQuebecAnnualError("Could not get hourly data")

        hourly_weather_data = []
        if not json_output.get('results'):
            # Missing Temperature data from Hydro-Quebec (but don't crash the app for that)
            hourly_weather_data = [None]*24
        else:
            hourly_weather_data = json_output['results'][0]['listeTemperaturesHeure']
        # Add temp in data
        processed_hourly_data = [{'hour': data['heure'],
                                  'lower': data['consoReg'],
                                  'high': data['consoHaut'],
                                  'total': data['consoTotal'],
                                  'temp': hourly_weather_data[i]}
                                 for i, data in enumerate(hourly_consumption_data)]

        raw_hourly_data = {'Energy': hourly_consumption_data,
                           'Power': hourly_power_data,
                           'Weather': hourly_weather_data}
        hourly_data = {'processed_hourly_data': processed_hourly_data,
                       'raw_hourly_data': raw_hourly_data}
        return hourly_data