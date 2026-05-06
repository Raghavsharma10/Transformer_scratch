def post(cls, payload):
        """
        A wrapper over Model.post() that handles the case where a Library has a PairedBarcode
        and the user may have supplied the PairedBarcode in the form of index1-index2, i.e. 
        GATTTCCA-GGCGTCGA. This isn't the PairedBarcode's record name or a record ID, thus 
        Model.post() won't be able to figure out the PairedBarcode's ID to substitute in the payload
        (via a call to cls.replace_name_with_id()). Thus, this wrapper will attempt to replace
        a PairedBarcode sequence in the payload with a PairedBarcode ID, then pass the payload off
        to Model.post().
        """
        slpk_attr_name = "sequencing_library_prep_kit_id"
        paired_bc_id_attr_name = "paired_barcode_id"
        seq_reg = re.compile("^[ACGTN]+$")
        if paired_bc_id_attr_name in payload:
            try:
                index1, index2 = payload[paired_bc_id_attr_name].upper().split("-")
            except ValueError:
                # Not in GATTTCCA-GGCGTCGA format so let it be. 
                return Model.post(cls=cls, payload=payload)
            if not seq_reg.match(index1) or not seq_reg.match(index2):
                # Not in GATTTCCA-GGCGTCGA format so let it be. 
                return Model.post(cls=cls, payload=payload)
            if not slpk_attr_name in payload:
                raise Exception("You need to include the " + slpk + " attribute name.")
            slpk_id = SequencingLibraryPrepKit.replace_name_with_id(payload[slpk_attr_name])
            payload[slpk_attr_name] = slpk_id
           
            index1_id = Barcode.find_by(payload={slpk_attr_name: slpk_id, "index_number": 1, "sequence": index1}, require=True)["id"]
            index2_id = Barcode.find_by(payload={slpk_attr_name: slpk_id, "index_number": 2, "sequence": index2}, require=True)["id"]
            # Ensure that PairedBarcode for this index combo already exists:
            pbc_payload = {"index1_id": index1_id, "index2_id": index2_id, slpk_attr_name: slpk_id}
            pbc_exists = PairedBarcode.find_by(payload=pbc_payload)
            if not pbc_exists:
                pbc_exists = PairedBarcode.post(payload=pbc_payload)
            pbc_id = pbc_exists["id"]
            payload[paired_bc_id_attr_name] = pbc_id
        return super().post(payload=payload)