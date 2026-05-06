def get_content_string(self):
        """ Ge thet Clusterpoint response's content as a string. """
        return ''.join([ET.tostring(element, encoding="utf-8", method="xml")
                        for element in list(self._content)])