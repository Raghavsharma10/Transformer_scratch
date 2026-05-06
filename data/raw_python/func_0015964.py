def on_command_control(self, val_id):
        """
        Parameters
        ----------
        val_id : str

        Returns
        -------
        requests.Response
        """
        data = "control,controlId=1|" + val_id
        return self._basic_post(url='commandControlPublic', data=data)