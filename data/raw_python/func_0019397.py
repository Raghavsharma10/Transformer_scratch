def get_game_logs(self):
        """Returns team game logs as a pandas DataFrame"""
        logs = self.response.json()['resultSets'][0]['rowSet']
        headers = self.response.json()['resultSets'][0]['headers']
        df = pd.DataFrame(logs, columns=headers)
        df.GAME_DATE = pd.to_datetime(df.GAME_DATE)
        return df