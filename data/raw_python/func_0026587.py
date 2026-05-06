def _reset_flood_offenders(self, *args):
        """Resets the list of flood offenders on event trigger"""

        offenders = []
        # self.log('Resetting flood offenders')

        for offender, offence_time in self._flooding.items():
            if time() - offence_time < 10:
                self.log('Removed offender from flood list:', offender)
                offenders.append(offender)

        for offender in offenders:
            del self._flooding[offender]