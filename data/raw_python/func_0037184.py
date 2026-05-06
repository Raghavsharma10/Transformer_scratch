def uploadFile(self, file_path, upload_path = '', overwrite = False):
        """uploadFile

        Remarks:
            How Ndrive uploads a file to its server:
                1. POST /CheckStatus.ndrive
                2. POST /GetDiskSpace.ndrive
                3. POST /CheckUpload.ndrive
                4. PUT /FILE_PATH
                5. POST /GetList.ndrive
                6. POST /GetWasteInfo.ndrive
                7. POST /GetDiskSpace.ndrive
        """
        s = self.checkStatus()
        s = self.getDiskSpace()
        s = self.checkUpload(file_path, upload_path, overwrite)

        if s is True:
            self.put(file_path, upload_path)