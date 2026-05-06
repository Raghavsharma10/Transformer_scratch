def insertFiles(self, filesList, qInserts=False):
        """
        API to insert a list of file into DBS in DBS. Up to 10 files can be inserted in one request.

        :param qInserts: True means that inserts will be queued instead of done immediately. INSERT QUEUE Manager will perform the inserts, within few minutes.
        :type qInserts: bool
        :param filesList: List of dictionaries containing following information
        :type filesList: list of dicts
        :key logical_file_name: File to be inserted (str) (Required)
        :key is_file_valid: (optional, default = 1): (bool)
        :key block: required: /a/b/c#d (str)
        :key dataset: required: /a/b/c (str)
        :key file_type: (optional, default = EDM) one of the predefined types, (str)
        :key check_sum: (optional, default = '-1') (str)
        :key event_count: (optional, default = -1) (int)
        :key file_size: (optional, default = -1.) (float)
        :key adler32: (optional, default = '') (str)
        :key md5: (optional, default = '') (str)
        :key auto_cross_section: (optional, default = -1.) (float)
        :key file_lumi_list: (optional, default = []) [{'run_num': 123, 'lumi_section_num': 12},{}....]
        :key file_parent_list: (optional, default = []) [{'file_parent_lfn': 'mylfn'},{}....]
        :key file_assoc_list: (optional, default = []) [{'file_parent_lfn': 'mylfn'},{}....]
        :key file_output_config_list: (optional, default = []) [{'app_name':..., 'release_version':..., 'pset_hash':...., output_module_label':...},{}.....]

        """

        if not qInserts: #turn off qInserts
            return self.__callServer("files", params={'qInserts': qInserts}, data=filesList, callmethod='POST' )
        return self.__callServer("files", data=filesList, callmethod='POST' )