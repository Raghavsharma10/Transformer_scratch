def _did_create_child(self, connection):
        """ Callback called after adding a new child nurest_object """

        response = connection.response
        try:
            connection.user_info['nurest_object'].from_dict(response.data[0])
        except Exception:
            pass

        return self._did_perform_standard_operation(connection)