def query(self, query, **kwargs):
        """Send AMCP command"""
        if not self.connection:
            if not self.connect():
                return CasparResponse(500, "Unable to connect CasparCG server")

        query = query.strip()
        if kwargs.get("verbose", True):
            if not query.startswith("INFO"):
                logging.debug("Executing AMCP: {}".format(query))
        query += "\r\n"

        if PYTHON_VERSION >= 3:
            query = bytes(query.encode("utf-8"))
            delim = bytes("\r\n".encode("utf-8"))
        else:
            delim = "\r\n"

        try:
            self.connection.write(query)
            result = self.connection.read_until(delim).strip()
        except Exception:
            log_traceback()
            return CasparResponse(500, "Query failed")

        if PYTHON_VERSION >= 3:
            result = result.decode("UTF-8")

        if not result:
            return CasparResponse(500, "No result")

        try:
            if result[0:3] == "202":
                return CasparResponse(202, "No result")

            elif result[0:3] in ["201", "200"]:
                stat = int(result[0:3])
                result = decode_if_py3(self.connection.read_until(delim)).strip()
                return CasparResponse(stat, result)

            elif int(result[0:1]) > 3:
                stat = int(result[0:3])
                return CasparResponse(stat, result)
        except Exception:
            log_traceback()
            return CasparResponse(500, "Malformed result: {}".format(result))
        return CasparResponse(500, "Unexpected result: {}".format(result))