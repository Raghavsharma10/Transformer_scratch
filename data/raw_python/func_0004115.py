def create_from_name_and_dictionary(self, name, datas):
        """Return a populated object Parameter from dictionary datas
        """
        parameter = ObjectParameter()
        self.set_common_datas(parameter, name, datas)

        if "optional" in datas:
            parameter.optional = to_boolean(datas["optional"])
        if "type" in datas:
            parameter.type = str(datas["type"])
        if "generic" in datas:
            parameter.generic = to_boolean(datas["generic"])

        return parameter