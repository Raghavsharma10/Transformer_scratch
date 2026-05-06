def save(self, name=None, path=None):
        """Save file as xml
        """
        if path :
            name = os.path.join(path,name)

        try:
            self._create_table_xml_file(self.etree, name)
        except (Exception,) as e:
            print(e)
            return False

        return True