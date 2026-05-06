def stop_host(self, config_file):
        """
        Stops a managed host specified by `config_file`.
        """
        res = self.send_json_request('host/stop', data={'config': config_file})

        if res.status_code != 200:
            raise UnexpectedResponse(
                'Attempted to stop a JSHost. Response: {res_code}: {res_text}'.format(
                    res_code=res.status_code,
                    res_text=res.text,
                )
            )

        return res.json()