def stop(self):
        """
        The client stop method. If the client is currently connected
        stop the thread and disconnect. Publish the disconnected
        message if clean shutdown.
        """
        # self.loop.call_soon_threadsafe(self.loop.stop)
        # self.loop.stop()
        # self._longPoll.cancel()
        # self._shortPoll.cancel()
        if self.connected:
            LOGGER.info('Disconnecting from MQTT... {}:{}'.format(self._server, self._port))
            self._mqttc.publish(self.topicSelfConnection, json.dumps({'node': self.profileNum, 'connected': False}), retain=True)
            self._mqttc.loop_stop()
            self._mqttc.disconnect()
        try:
            for watcher in self.__stopObservers:
                watcher()
        except KeyError as e:
            LOGGER.exception('KeyError in gotConfig: {}'.format(e), exc_info=True)