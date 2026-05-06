def publish(self, topic="/controller", qos=0, payload=None):
        """
        publish(self, topic, payload=None, qos=0, retain=False)
        Returns a tuple (result, mid), where result is MQTT_ERR_SUCCESS to
        indicate success or MQTT_ERR_NO_CONN if the client is not currently
        connected.  mid is the message ID for the publish request. The mid
        value can be used to track the publish request by checking against the
        mid argument in the on_publish() callback if it is defined.
        """
        result = self.client.publish(topic,
                                     payload=json.dumps(payload),
                                     qos=qos)
        if result[0] == mqtt.MQTT_ERR_NO_CONN:
            raise RuntimeError("No connection")
        return result[1]