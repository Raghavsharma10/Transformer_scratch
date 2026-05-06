def nodes_for_spec(self, spec):
        """
            Determine nodes for an input_algorithms spec
            Taking into account nested specs
        """
        tokens = []
        if isinstance(spec, sb.create_spec):
            container = nodes.container(classes=["option_spec_option shortline blue-back"])
            creates = spec.kls
            for name, option in sorted(spec.kwargs.items(), key=lambda x: len(x[0])):
                para = nodes.paragraph(classes=["option monospaced"])
                para += nodes.Text("{0} = ".format(name))
                self.nodes_for_signature(option, para)

                fields = {}
                if creates and hasattr(creates, 'fields') and isinstance(creates.fields, dict):
                    for key, val in creates.fields.items():
                        if isinstance(key, tuple):
                            fields[key[0]] = val
                        else:
                            fields[key] = val

                txt = fields.get(name) or "No description"
                viewlist = ViewList()
                for line in dedent(txt).split('\n'):
                    viewlist.append(line, name)
                desc = nodes.section(classes=["description monospaced"])
                self.state.nested_parse(viewlist, self.content_offset, desc)

                container += para
                container += desc
                container.extend(self.nodes_for_spec(option))
            tokens.append(container)
        elif isinstance(spec, sb.optional_spec):
            tokens.extend(self.nodes_for_spec(spec.spec))
        elif isinstance(spec, sb.container_spec):
            tokens.extend(self.nodes_for_spec(spec.spec))
        elif isinstance(spec, sb.dictof):
            tokens.extend(self.nodes_for_spec(spec.value_spec))

        return tokens