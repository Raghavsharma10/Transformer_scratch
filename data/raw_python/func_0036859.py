def get_load_status(self):
        """
        To check the load status of a service.
        :return: a dictionary that include json data.
                 example: { "status": "ok", "running": "0", "pending": "0", "finished": "0", "node_name": "node-name" }
        """
        url, method = self.command_set['daemonstatus'][0], self.command_set['daemonstatus'][1]
        response = http_utils.request(url, method_type=method, return_type=http_utils.RETURN_JSON)
        if response is None:
            logging.warning('%s failure: not found or connection fail' % sys._getframe().f_code.co_name)
            response = DaemonStatus().__dict__
        return response