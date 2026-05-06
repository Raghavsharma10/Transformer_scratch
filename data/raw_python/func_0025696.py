def put(self, targetId):
        """
        stores a new target.
        :param targetId: the target to store.
        :return:
        """
        json = request.get_json()
        if 'hinge' in json:
            logger.info('Storing target ' + targetId)
            if self._targetController.storeFromHinge(targetId, json['hinge']):
                logger.info('Stored target ' + targetId)
                return None, 200
            else:
                return None, 500
        else:
            return None, 400