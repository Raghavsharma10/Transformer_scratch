def uploadFile(self, file_obj, full_path, overwrite = False):
        """Upload a file as Ndrive really do.

            >>> nd.uploadFile('~/flower.png','/Picture/flower.png',True)

        This function imitates the process when Ndrive uploads a local file to its server. The process follows 7 steps:
              1. POST /CheckStatus.ndrive
              2. POST /GetDiskSpace.ndrive
              3. POST /CheckUpload.ndrive
              4. PUT /FILE_PATH
              5. POST /GetList.ndrive
              6. POST /GetWasteInfo.ndrive
              7. POST /GetDiskSpace.ndrive

            nd.uploadFile('./flower.png','/Picture/flower.png')

        :param file_obj: A file-like object to check whether possible to upload. You can pass a string as a file_obj or a real file object.
        :param full_path: The full path to upload the file to, *including the file name*. If the destination directory does not yet exist, it will be created. 
        :param overwrite: Whether to overwrite an existing file at the given path. (Default ``False``.)
        """
        s = self.checkStatus()
        s = self.getDiskSpace()
        s = self.checkUpload(file_obj, full_path, overwrite)

        if s is True:
            self.put(file_obj, full_path, overwrite)