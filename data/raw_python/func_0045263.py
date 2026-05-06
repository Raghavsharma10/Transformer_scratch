async def set_start_date(self, date: str, time: str, check_in_duration: int = None):
        """ set the tournament start date (and check in duration)

        |methcoro|

        Args:
            date: fomatted date as YYYY/MM/DD (2017/02/14)
            time: fromatted time as HH:MM (20:15)
            check_in_duration (optional): duration in minutes

        Raises:
            APIException

        """
        date_time = datetime.strptime(date + ' ' + time, '%Y/%m/%d %H:%M')
        res = await self.connection('PUT',
                                    'tournaments/{}'.format(self._id),
                                    'tournament',
                                    start_at=date_time,
                                    check_in_duration=check_in_duration or 0)
        self._refresh_from_json(res)