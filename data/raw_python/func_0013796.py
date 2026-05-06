def _is_compress_filetype(self, inpath):
        """private method that performs magic number and size check on file to determine whether to compress the file"""
        # check for common file type suffixes in order to avoid the need for file reads to check magic number for binary vs. text file
        if self._is_common_binary(inpath):
            return False
        elif self._is_common_text(inpath):
            return True
        else:
            # files > 10kB get checked for compression (arbitrary decision to skip compression on small files)
            the_file_size = file_size(inpath)
            if the_file_size > 10240:
                if the_file_size > 512000:  # seems to be a break point at ~ 500kb where file compression offset by additional file read, so limit tests to files > 500kB
                    try:
                        system_command = "file --mime-type -b " + quote(inpath)
                        response = muterun(system_command)
                        if response.stdout[0:5] == "text/":  # check for a text file mime type
                            return True   # appropriate size, appropriate file mime type
                        else:
                            return False  # appropriate size, inappropriate file mime type
                    except Exception:
                        return False
                else:
                    return True  # if file size is < 500kB, skip the additional file read and just go with compression
            else:
                return False