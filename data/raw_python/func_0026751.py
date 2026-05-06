def save(self, bury=False, final=False):
        """Write entry to JSON file in the proper location.

        Arguments
        ---------
        bury : bool

        final : bool
            If this is the 'final' save, perform additional sanitization and
            cleaning operations.

        """
        outdir, filename = self._get_save_path(bury=bury)

        if final:
            self.sanitize()

        # FIX: use 'dump' not 'dumps'
        jsonstring = json.dumps(
            {
                self[self._KEYS.NAME]: self._ordered(self)
            },
            indent='\t' if sys.version_info[0] >= 3 else 4,
            separators=(',', ':'),
            ensure_ascii=False)
        if not os.path.isdir(outdir):
            raise RuntimeError("Output directory '{}' for event '{}' does "
                               "not exist.".format(outdir, self[
                                   self._KEYS.NAME]))
        save_name = os.path.join(outdir, filename + '.json')
        with codecs.open(save_name, 'w', encoding='utf8') as sf:
            sf.write(jsonstring)

        if not os.path.exists(save_name):
            raise RuntimeError("File '{}' was not saved!".format(save_name))

        return save_name