def run(self):
        """For each file in noseOfYeti/specs, output nodes to represent each spec file"""
        tokens = []
        for name, spec in (("Harpoon", HarpoonSpec().harpoon_spec), ("Image", HarpoonSpec().image_spec)):
            section = nodes.section()
            section['names'].append(name)
            section['ids'].append(name)

            header = nodes.title()
            header += nodes.Text(name)
            section.append(header)

            section.extend(self.nodes_for_spec(spec))
            tokens.append(section)

        return tokens