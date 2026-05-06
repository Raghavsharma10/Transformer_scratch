def stop(self):
        """
        If the manager is running, tell it to stop its process
        """
        res = self.send_request('manager/stop', post=True)

        if res.status_code != 200:
            raise UnexpectedResponse(
                'Attempted to stop manager. {res_code}: {res_text}'.format(
                    res_code=res.status_code,
                    res_text=res.text,
                )
            )

        if settings.VERBOSITY >= verbosity.PROCESS_STOP:
            print('Stopped {}'.format(self.get_name()))

        # The request will end just before the process stops, so there is a tiny
        # possibility of a race condition. We delay as a precaution so that we
        # can be reasonably confident of the system's state.
        time.sleep(0.05)