def _make_spec_file(self):
        """Generates the text of an RPM spec file.

        Returns:
          A list of strings containing the lines of text.
        """
        # Note that bdist_rpm can be an old style class.
        if issubclass(BdistRPMCommand, object):
            spec_file = super(BdistRPMCommand, self)._make_spec_file()
        else:
            spec_file = bdist_rpm._make_spec_file(self)

        if sys.version_info[0] < 3:
            python_package = "python"
        else:
            python_package = "python3"

        description = []
        summary = ""
        in_description = False

        python_spec_file = []
        for line in spec_file:
            if line.startswith("Summary: "):
                summary = line

            elif line.startswith("BuildRequires: "):
                line = "BuildRequires: {0:s}-setuptools".format(python_package)

            elif line.startswith("Requires: "):
                if python_package == "python3":
                    line = line.replace("python", "python3")

            elif line.startswith("%description"):
                in_description = True

            elif line.startswith("%files"):
                line = "%files -f INSTALLED_FILES -n {0:s}-%{{name}}".format(
                    python_package)

            elif line.startswith("%prep"):
                in_description = False

                python_spec_file.append(
                    "%package -n {0:s}-%{{name}}".format(python_package))
                python_spec_file.append("{0:s}".format(summary))
                python_spec_file.append("")
                python_spec_file.append(
                    "%description -n {0:s}-%{{name}}".format(python_package))
                python_spec_file.extend(description)

            elif in_description:
                # Ignore leading white lines in the description.
                if not description and not line:
                    continue

                description.append(line)

            python_spec_file.append(line)

        return python_spec_file