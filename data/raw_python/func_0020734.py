def insertFile(self, qInserts=False):
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
        if qInserts in (False, 'False'): qInserts=False
        try:
            body = request.body.read()
            indata = cjson.decode(body)["files"]
            if not isinstance(indata, (list, dict)):
                dbsExceptionHandler("dbsException-invalid-input", "Invalid Input DataType", self.logger.exception, \
                                      "insertFile expects input as list or dirc")
            businput = []
            if isinstance(indata, dict):
                indata = [indata]
            indata = validateJSONInputNoCopy("files", indata)
            for f in indata:
                f.update({
                     #"dataset":f["dataset"],
                     "creation_date": f.get("creation_date", dbsUtils().getTime()),
                     "create_by" : dbsUtils().getCreateBy(),
                     "last_modification_date": f.get("last_modification_date", dbsUtils().getTime()),
                     "last_modified_by": f.get("last_modified_by", dbsUtils().getCreateBy()),
                     "file_lumi_list":f.get("file_lumi_list", []),
                     "file_parent_list":f.get("file_parent_list", []),
                     "file_assoc_list":f.get("assoc_list", []),
                     "file_output_config_list":f.get("file_output_config_list", [])})
                businput.append(f)
            self.dbsFile.insertFile(businput, qInserts)
        except cjson.DecodeError as dc:
            dbsExceptionHandler("dbsException-invalid-input2", "Wrong format/data from insert File input",  self.logger.exception, str(dc))
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.message)
        except HTTPError as he:
            raise he
        except Exception as ex:
            sError = "DBSWriterModel/insertFile. %s\n. Exception trace: \n %s" \
                    % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error',  dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)