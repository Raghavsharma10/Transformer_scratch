def patch(self):
        """
        Allows the UI to update parameters ensuring that all devices are kept in sync. Payload is json in TargetState
        format.
        :return:
        """
        # TODO block until all devices have updated?
        json = request.get_json()
        logger.info("Updating target state with " + str(json))
        self._targetStateController.updateTargetState(json)
        return None, 200