def update_status(self, status):
        """
        Updates the status of the deal

        :param status: status have to be ('won', 'pending', 'lost')
        :return: successfull response or raise Exception
        :rtype:
        """
        assert (status in (HightonConstants.WON, HightonConstants.PENDING, HightonConstants.LOST))
        from highton.models import Status

        status_obj = Status(name=status)
        return self._put_request(
            data=status_obj.element_to_string(status_obj.encode()),
            endpoint=self.ENDPOINT + '/' + str(self.id) + '/status',
        )