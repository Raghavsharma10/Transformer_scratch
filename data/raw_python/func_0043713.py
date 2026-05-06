def set_source_variable(self, source_id, variable, value):
        """ Change the value of a source variable. """
        source_id = int(source_id)
        return self._send_cmd("SET S[%d].%s=\"%s\"" % (
            source_id, variable, value))