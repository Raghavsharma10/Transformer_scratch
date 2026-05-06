def build_response(
            self,
            status=NOT_SET,
            error="",
            data=None):
        """build_response

        :param status: status code
        :param error: error message
        :param data: dictionary to send back
        """

        res_node = {
            "status": status,
            "error": error,
            "data": data
        }
        return res_node