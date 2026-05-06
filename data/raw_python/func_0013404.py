def _add_to_ref(self, rec_curr, line, lnum):
        """Add new fields to the current reference."""
        # Written by DV Klopfenstein
        # Examples of record lines containing ':' include:
        #   id: GO:0000002
        #   name: mitochondrial genome maintenance
        #   namespace: biological_process
        #   def: "The maintenance of ...
        #   is_a: GO:0007005 ! mitochondrion organization
        mtch = re.match(r'^(\S+):\s*(\S.*)$', line)
        if mtch:
            field_name = mtch.group(1)
            field_value = mtch.group(2)
            if field_name == "id":
                self._chk_none(rec_curr.id, lnum)
                rec_curr.id = field_value
            elif field_name == "alt_id":
                rec_curr.alt_ids.append(field_value)
            elif field_name == "name":
                self._chk_none(rec_curr.name, lnum)
                rec_curr.name = field_value
            elif field_name == "namespace":
                self._chk_none(rec_curr.namespace, lnum)
                rec_curr.namespace = field_value
            elif field_name == "is_a":
                rec_curr._parents.append(field_value.split()[0])
            elif field_name == "is_obsolete" and field_value == "true":
                rec_curr.is_obsolete = True
            elif field_name in self.optional_attrs:
                self.update_rec(rec_curr, field_name, field_value)
        else:
            self._die("UNEXPECTED FIELD CONTENT: {L}\n".format(L=line), lnum)