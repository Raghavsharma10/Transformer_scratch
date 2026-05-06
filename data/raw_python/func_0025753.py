def get(self):
        """
        Reloads the measurements from the backing store.
        :return: 200 if success.
        """
        try:
            self._measurementController.reloadCompletedMeasurements()
            return None, 200
        except:
            logger.exception("Failed to reload measurements")
            return str(sys.exc_info()), 500