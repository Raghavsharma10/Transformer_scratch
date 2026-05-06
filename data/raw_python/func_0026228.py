def sendHeartbeat(self):
        """
        Posts the current state to the server.
        :param serverURL: the URL to ping.
        :return:
        """
        for name, md in self.cfg.recordingDevices.items():
            try:
                data = marshal(md, recordingDeviceFields)
                data['serviceURL'] = self.cfg.getServiceURL() + API_PREFIX + '/devices/' + name
                targetURL = self.serverURL + API_PREFIX + '/devices/' + name
                logger.info("Pinging " + targetURL)
                resp = self.httpclient.put(targetURL, json=data)
                if resp.status_code != 200:
                    logger.warning("Unable to ping server at " + targetURL + " with " + str(data.keys()) +
                                   ", response is " + str(resp.status_code))
                else:
                    logger.info("Pinged server at " + targetURL + " with " + str(data.items()))
            except:
                logger.exception("Unable to ping server")