def delete_command_control_timers(self, val_id):
        """
        Parameters
        ----------
        val_id : str

        Returns
        -------
        requests.Response
        """
        data = "deleteTimers,controlId=" + val_id
        return self._basic_post(url='commandControlPublic', data=data)