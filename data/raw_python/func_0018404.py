def get_instrument_history(self, instrument, candle_format="bidask",
                               granularity='S5', count=500,
                               daily_alignment=None, alignment_timezone=None,
                               weekly_alignment="Monday", start=None,
                               end=None):
        """
            See more:
            http://developer.oanda.com/rest-live/rates/#retrieveInstrumentHistory
        """
        url = "{0}/{1}/candles".format(self.domain, self.API_VERSION)
        params = {
            "accountId": self.account_id,
            "instrument": instrument,
            "candleFormat": candle_format,
            "granularity": granularity,
            "count": count,
            "dailyAlignment": daily_alignment,
            "alignmentTimezone": alignment_timezone,
            "weeklyAlignment": weekly_alignment,
            "start": start,
            "end": end,
        }
        try:
            return self._Client__call(uri=url, params=params, method="get")
        except RequestException:
            return False
        except AssertionError:
            return False