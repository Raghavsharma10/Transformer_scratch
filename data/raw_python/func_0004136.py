def set_common_datas(self, element, name, datas):
        """Populated common data for an element from dictionnary datas
        """
        element.name = str(name)
        if "description" in datas:
            element.description = str(datas["description"]).strip()

        if isinstance(element, Sampleable) and element.sample is None and "sample" in datas:
            element.sample = str(datas["sample"]).strip()

        if isinstance(element, Displayable):
            if "display" in datas:
                element.display = to_boolean(datas["display"])

            if "label" in datas:
                element.label = datas["label"]
            else:
                element.label = element.name