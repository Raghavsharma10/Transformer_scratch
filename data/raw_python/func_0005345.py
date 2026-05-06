def upload(cls, path, document_type, is_protocol, description=""):
        """
        Args:
            path: `str`. The path to the document to upload. 
            document_type: `str`. DocumentType identified by the value of its name attribute. 
            is_protocol: `bool`. 
            description: `str`. 
        """
        file_name = os.path.basename(path)
        mime_type = mimetypes.guess_type(file_name)[0]
        data = base64.b64encode(open(path, 'rb').read())
        temp_uri = str(data, "utf-8")
        #href = "data:{mime_type};base64,{temp_uri}".format(mime_type=mime_type, temp_uri=temp_uri) 
        payload = {}
        payload["content_type"] = mime_type 
        payload["data"] = temp_uri
        payload["description"] = description
        payload["document_type_id"] = DocumentType(document_type).id
        payload["name"] =  file_name
        payload["is_protocol"] = is_protocol
        cls.post(payload)