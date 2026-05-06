def get(self, targetId):
        """
        Yields the analysed wav data.
        :param targetId: 
        :return: 
        """
        result = self._targetController.analyse(targetId)
        if result:
            if len(result) == 2:
                if result[1] == 404:
                    return result
                else:
                    return {'name': targetId, 'data': self._jsonify(result)}, 200
            else:
                return None, 404
        else:
            return None, 500