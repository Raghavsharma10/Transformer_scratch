def delete_datapoint(self, datapoint):
        """Delete the provided datapoint from this stream

        :raises devicecloud.DeviceCloudHttpException: in the case of an unexpected http error

        """
        datapoint = validate_type(datapoint, DataPoint)
        self._conn.delete("/ws/DataPoint/{stream_id}/{datapoint_id}".format(
            stream_id=self.get_stream_id(),
            datapoint_id=datapoint.get_id(),
        ))