def get_library_barcode_sequence_hash(self, inverse=False):
        """
        Calls the SequencingRequest's get_library_barcode_sequence_hash server-side endpoint to
        create a hash of the form {LibraryID -> barcode_sequence} for all Libraries on the 
        SequencingRequest. 

        Args:
            inverse: `bool`. True means to inverse the key and value pairs such that the barcode
                sequence serves as the key.

        Returns: `dict`.
        """
        action = os.path.join(self.record_url, "get_library_barcode_sequence_hash")
        res = requests.get(url=action, headers=HEADERS, verify=False)
        res.raise_for_status()
        res_json = res.json()
        # Convert library ID from string to int
        new_res = {}
        for lib_id in res_json:
            new_res[int(lib_id)] = res_json[lib_id]
        res_json = new_res

        if inverse:
            rev = {}
            for lib_id in res_json:
                rev[res_json[lib_id]] = lib_id
        res_json = rev
        return res_json