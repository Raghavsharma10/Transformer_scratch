def get_contents(self, folder: Folder):
        """
        List all contents of a folder. Returns a list of all Documents and Folders (in this order) in the folder.
        """
        log.debug("Listing Contents of %s/%s" % (folder.course.id, folder.id))
        if isinstance(folder, Course):
            response = json.loads(self._get('/api/documents/%s/folder' % folder.course.id).text)
        else:
            response = json.loads(self._get('/api/documents/%s/folder/%s' % (folder.course.id, folder.id)).text)
            log.debug("Got response: %s" % response)

        documents = [Document.from_response(response, folder) for response in response["documents"]]

        folders = [Folder.from_response(response, folder) for response in response["folders"]]

        return documents + folders