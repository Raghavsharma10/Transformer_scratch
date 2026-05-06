def get_object_info(self):
        """
        Returns object info in following form <module.class object at address>
        """
        objectinfo = str(self.__class__).replace(">", "")
        objectinfo = objectinfo.replace("class ", "")
        objectinfo = objectinfo.replace("'", "")
        objectinfo += " object at 0x%x>" % id(self)
        return objectinfo