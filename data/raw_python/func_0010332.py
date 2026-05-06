def delete(self):
        """Delete this stream from Device Cloud along with its history

        This call will return None on success and raise an exception in the event of an error
        performing the deletion.

        :raises devicecloud.DeviceCloudHttpException: in the case of an unexpected http error
        :raises devicecloud.streams.NoSuchStreamException: if this stream has already been deleted

        """
        try:
            self._conn.delete("/ws/DataStream/{}".format(self.get_stream_id()))
        except DeviceCloudHttpException as http_excpeption:
            if http_excpeption.response.status_code == 404:
                raise NoSuchStreamException()  # this branch is present, but the DC appears to just return 200 again
            else:
                raise http_excpeption