def delete_command_control(self, val_id):
        """
        Parameters
        ----------
        val_id : str

        Returns
        -------
        requests.Response
        """

        data = "delete,controlId=" + val_id
        return self._basic_post(url='commandControlPublic', data=data)