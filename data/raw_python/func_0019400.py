def get_shots(self):
        """Returns the shot chart data as a pandas DataFrame."""
        shots = self.response.json()['resultSets'][0]['rowSet']
        headers = self.response.json()['resultSets'][0]['headers']
        return pd.DataFrame(shots, columns=headers)