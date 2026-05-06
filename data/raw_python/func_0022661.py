def extract_yang_model(self, content):
        """
        Extracts one or more YANG models from an RFC or draft text string in
        which the models are specified. The function skips over page
        formatting (Page Headers and Footers) and performs basic YANG module
        syntax checking. In strict mode, the function also enforces the
        <CODE BEGINS> / <CODE ENDS> tags - a model is not extracted unless
        the tags are present.
        :return: None
        """
        model = []
        output_file = None
        in_model = False
        example_match = False
        i = 0
        level = 0
        quotes = 0
        while i < len(content):
            line = content[i]

            # Try to match '<CODE ENDS>'
            if self.CODE_ENDS_TAG.match(line):
                if in_model is False:
                    self.warning("Line %d: misplaced <CODE ENDS>" % i)
                in_model = False

            if "\"" in line:
                if line.count("\"") % 2 == 0:
                    quotes = 0
                else:
                    if quotes == 1:
                        quotes = 0
                    else:
                        quotes = 1

            # Try to match '(sub)module <module_name> {'
            match = self.MODULE_STATEMENT.match(line)
            if match:
                # We're already parsing a module
                if quotes == 0:
                    if level > 0:
                        self.error("Line %d - 'module' statement within another module" % i)
                        return

                # Check if we should enforce <CODE BEGINS> / <CODE ENDS>
                # if we do enforce, we ignore models  not enclosed in <CODE BEGINS> / <CODE ENDS>
                if match.groups()[1] or match.groups()[4]:
                    self.warning('Line %d - Module name should not be enclosed in quotes' % i)

                # do the module name checking, etc.
                example_match = self.EXAMPLE_TAG.match(match.groups()[2])
                if in_model is True:
                    if example_match:
                        self.error("Line %d - YANG module '%s' with <CODE BEGINS> and starting with 'example-'" %
                                   (i, match.groups()[2]))
                else:
                    if not example_match:
                        self.error("Line %d - YANG module '%s' with no <CODE BEGINS> and not starting with 'example-'" %
                                   (i, match.groups()[2]))

                # now decide if we're allowed to set the level
                # (i.e. signal that we're in a module) to 1 and if
                # we're allowed to output the module at all with the
                # strict examples flag
                # if self.strict is True:
                #     if in_model is True:
                #         level = 1
                # else:
                #     level = 1

                # always set the level to 1; we decide whether or not
                # to output at the end
                if quotes == 0:
                    level = 1
                if not output_file and level == 1 and quotes == 0:
                    print("\nExtracting '%s'" % match.groups()[2])
                    output_file = '%s.yang' % match.groups()[2].strip('"\'')
                    if self.debug_level > 0:
                        print('   Getting YANG file name from module name: %s' % output_file)

            if level > 0:
                self.debug_print_line(i, level, content[i])
                # Try to match the Footer ('[Page <page_num>]')
                # If match found, skip over page headers and footers
                if self.PAGE_TAG.match(line):
                    self.strip_empty_lines_backward(model, 3)
                    self.debug_print_strip_msg(i, content[i])
                    i += 1        # Strip the
                    # Strip empty lines between the Footer and the next page Header
                    i = self.strip_empty_lines_forward(content, i)
                    if i < len(content):
                        self.debug_print_strip_msg(i, content[i])
                        i += 1      # Strip the next page Header
                    else:
                        self.error("<End of File> - EOF encountered while parsing the model")
                        return
                    # Strip empty lines between the page Header and real content on the page
                    i = self.strip_empty_lines_forward(content, i) - 1
                    if i >= len(content):
                        self.error("<End of File> - EOF encountered while parsing the model")
                        return
                else:
                    model.append([line, i + 1])
                    counter = Counter(line)
                    if quotes == 0:
                        if "\"" in line and "}" in line:
                            if line.index("}") > line.rindex("\"") or line.index("}") < line.index("\""):
                                level += (counter['{'] - counter['}'])
                        else:
                            level += (counter['{'] - counter['}'])
                    if level == 1:
                        if self.strict:
                            if self.strict_examples:
                                if example_match and not in_model:
                                    self.write_model_to_file(model, output_file)
                            elif in_model:
                                self.write_model_to_file(model, output_file)
                        else:
                            self.write_model_to_file(model, output_file)
                        self.max_line_len = 0
                        model = []
                        output_file = None
                        level = 0

            # Try to match '<CODE BEGINS>'
            match = self.CODE_BEGINS_TAG.match(line)
            if match:
                # Found the beginning of the YANG module code section; make sure we're not parsing a model already
                if level > 0:
                    self.error("Line %d - <CODE BEGINS> within a model" % i)
                    return
                if in_model is True:
                    self.error("Line %d - Misplaced <CODE BEGINS> or missing <CODE ENDS>" % i)
                in_model = True
                mg = match.groups()
                # Get the YANG module's file name
                if mg[2]:
                    print("\nExtracting '%s'" % match.groups()[2])
                    output_file = mg[2].strip()
                else:
                    if mg[0] and mg[1] is None:
                        self.error('Line %d - Missing file name in <CODE BEGINS>' % i)
                    else:
                        self.error("Line %d - YANG file not specified in <CODE BEGINS>" % i)
            i += 1
        if level > 0:
            self.error("<End of File> - EOF encountered while parsing the model")
            return
        if in_model is True:
            self.error("Line %d - Missing <CODE ENDS>" % i)