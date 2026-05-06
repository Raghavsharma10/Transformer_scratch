def as_list(self):
        """Returns GO annotation as a flat list (in GAF 2.1 format order)."""
        go_id = self.go_term.id
        
        qual_str = '|'.join(self.qualifier)
        db_ref_str = '|'.join(self.db_ref)
        taxon_str = '|'.join(self.taxon)

        # with_from is currently left as a string
        # with_from = '|'.join()
        with_from_str = self.with_from or ''
        db_name_str = self.db_name or ''
        db_syn_str = '|'.join(self.db_syn)
        ext_str = '|'.join(self.ext)
        product_id_str = self.product_id or ''

        l = [
            self.db,
            self.db_id,
            self.db_symbol,
            qual_str,
            go_id,
            db_ref_str,
            self.ev_code,
            with_from_str,
            self.aspect,
            db_name_str,
            db_syn_str,
            self.db_type,
            taxon_str,
            self.date,
            self.assigned_by,
            ext_str,
            product_id_str,
        ]
        return l