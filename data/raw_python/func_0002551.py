def _extract_yaml_block(self, indent, fh):
        """Extract a raw yaml block from a file handler"""
        raw_yaml = []
        indent_match = re.compile(r"^{}".format(indent))
        try:
            fh.next()
            while indent_match.match(fh.peek()):
                raw_yaml.append(fh.next().replace(indent, "", 1))
                # check for the end and stop adding yaml if encountered
                if self.yaml_block_end.match(fh.peek()):
                    fh.next()
                    break
        except StopIteration:
            pass
        return "\n".join(raw_yaml)