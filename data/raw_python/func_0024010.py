def append(self, filename_in_zip, file_contents):
        '''
        Appends a file with name filename_in_zip and contents of
        file_contents to the in-memory zip.
        '''
        # Set the file pointer to the end of the file
        self.in_memory_zip.seek(-1, io.SEEK_END)

        # Get a handle to the in-memory zip in append mode
        zf = zipfile.ZipFile(self.in_memory_zip, "a", zipfile.ZIP_DEFLATED, False)

        # Write the file to the in-memory zip
        zf.writestr(filename_in_zip, file_contents)

        # Mark the files as having been created on Windows so that
        # Unix permissions are not inferred as 0000
        for zfile in zf.filelist:
            zfile.create_system = 0

        # Close the ZipFile
        zf.close()

        # Rewind the file
        self.in_memory_zip.seek(0)

        return self