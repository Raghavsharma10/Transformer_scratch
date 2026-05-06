def write_yara(self, output_file):
        """
        Write out yara signatures to a file.
        """
        fout = open(output_file, 'wb')
        fout.write('\n')

        for iocid in self.yara_signatures:
            signature = self.yara_signatures[iocid]
            fout.write(signature)
            fout.write('\n')

        fout.close()
        return True