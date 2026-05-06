def write_model_to_file(self, mdl, fn):
        """
        Write a YANG model that was extracted from a source identifier
        (URL or source .txt file) to a .yang destination file
        :param mdl: YANG model, as a list of lines
        :param fn: Name of the YANG model file
        :return:
        """
        # Write the model to file
        output = ''.join(self.post_process_model(mdl, self.add_line_refs))
        if fn:
            fqfn = self.dst_dir + '/' + fn
            if os.path.isfile(fqfn):
                self.error("File '%s' exists" % fqfn)
                return
            with open(fqfn, 'w') as of:
                of.write(output)
                of.close()
                self.extracted_models.append(fn)
        else:
            self.error("Output file name can not be determined; YANG file not created")