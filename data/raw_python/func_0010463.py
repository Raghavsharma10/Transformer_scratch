def do_pot(self):
        """
        Sync the template with the python code.
        """
        files_to_translate = []
        log.debug("Collecting python sources for pot ...")
        for source_path in self._source_paths:
            for source_path in self._iter_suffix(path=source_path, suffix=".py"):
                log.debug("... add to pot: {source}".format(source=str(source_path)))
                files_to_translate.append(str(source_path))
        for system_file in self.SYSTEM_SOURCE_FILES:
            files_to_translate.append(str(self._system_path / system_file))
            # FIXME: use separate domain for system source translations? Nerge them when generating mo's?
        log.debug("Finished collection sources.")
        pot_path = (self._po_path / self._basename).with_suffix(".pot")
        command = ["xgettext", "--keyword=_", "--keyword=_translate",
                   "--output={output}".format(output=str(pot_path))]
        command.extend(files_to_translate)
        check_call(command)
        log.debug("pot file \"{pot}\" created!".format(pot=str(pot_path)))

        pot_copy_path = self._mo_path / pot_path.name
        log.debug("Copying pot file to mo path: {pot_copy_path}".format(pot_copy_path=str(pot_copy_path)))
        shutil.copy(str(pot_path), str(pot_copy_path))