def create_from_dictionary(self, datas):
        """Return a populated object ResponseCode from dictionary datas
        """
        if "code" not in datas:
            raise ValueError("A response code must contain a code in \"%s\"." % repr(datas))

        code = ObjectResponseCode()
        self.set_common_datas(code, str(datas["code"]), datas)

        code.code = int(datas["code"])
        if "message" in datas:
            code.message = str(datas["message"])
        elif code.code in self.default_messages.keys():
            code.message = self.default_messages[code.code]
        if "generic" in datas:
            code.generic = to_boolean(datas["generic"])

        return code