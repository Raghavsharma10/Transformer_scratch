def _submit_request(self):
        """Submit a request to the ACS Zeropoint Calculator.

        If an exception is raised during the request, an error message is
        given. Otherwise, the response is saved in the corresponding
        attribute.

        """
        try:
            self._response = urlopen(self._url)
        except URLError as e:
            msg = ('{}\n{}\nThe query failed! Please check your inputs. '
                   'If the error persists, submit a ticket to the '
                   'ACS Help Desk at hsthelp.stsci.edu with the error message '
                   'displayed above.'.format(str(e), self._msg_div))
            LOG.error(msg)
            self._failed = True
        else:
            self._failed = False