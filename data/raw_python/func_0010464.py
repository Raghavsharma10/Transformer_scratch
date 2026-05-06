def do_po(self):
        """
        Update all po files with the data in the pot reference file.
        """
        log.debug("Start updating po files ...")
        pot_path = (self._po_path / self._basename).with_suffix(".pot")
        for po_dir_path in self._iter_po_dir():
            po_path = (po_dir_path / self._basename).with_suffix(".po")
            if po_path.exists():
                log.debug("update {po}:".format(po=str(po_path)))
                check_call(["msgmerge", "-U", str(po_path), str(pot_path)])
            else:
                log.debug("create {po}:".format(po=str(po_path)))
                check_call(["msginit", "-i", str(pot_path), "-o", str(po_path), "--no-translator"])
            po_copy_path = self._mo_path / po_path.parent.name / po_path.name
            po_copy_path.parent.mkdir(exist_ok=True)
            log.debug("Copying po file to mo path: {po_copy_path}".format(po_copy_path=str(po_copy_path)))

            shutil.copy(str(po_path), str(po_copy_path))
        log.debug("All po files updated")