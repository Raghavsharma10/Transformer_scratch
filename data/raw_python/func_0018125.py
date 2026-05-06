def _check_inputs(self):
        """Check the inputs to ensure they are valid.

        Returns
        -------
        status : bool
            True if all inputs are valid, False if one is not.

        """
        valid_detector = True
        valid_filter = True
        valid_date = True
        # Determine the submitted detector is valid
        if self.detector not in self._valid_detectors:
            msg = ('{} is not a valid detector option.\n'
                   'Please choose one of the following:\n{}\n'
                   '{}'.format(self.detector,
                               '\n'.join(self._valid_detectors),
                               self._msg_div))
            LOG.error(msg)
            valid_detector = False

        # Determine if the submitted filter is valid
        if (self.filt is not None and valid_detector and
                self.filt not in self.valid_filters[self.detector]):
            msg = ('{} is not a valid filter for {}\n'
                   'Please choose one of the following:\n{}\n'
                   '{}'.format(self.filt, self.detector,
                               '\n'.join(self.valid_filters[self.detector]),
                               self._msg_div))
            LOG.error(msg)
            valid_filter = False

        # Determine if the submitted date is valid
        date_check = self._check_date()
        if date_check is not None:
            LOG.error('{}\n{}'.format(date_check, self._msg_div))
            valid_date = False

        if not valid_detector or not valid_filter or not valid_date:
            return False

        return True