def create_temp_file(self, suffix, content):
        """ 
        Creates file, because environment variables are by default escaped it
        encodes and then decodes them before write so \n etc. work correctly.
        """
        temp = tempfile.NamedTemporaryFile(suffix=suffix)
        temp.write(content.encode('latin1').decode('unicode_escape').encode('utf-8'))
        temp.seek(0) # Resets the temp file line to 0
        return temp