def read_line(line):
        """Reads lines of XML and delimits, strips, and returns."""
        name, value = '', ''

        if '=' in line:
            name, value = line.split('=', 1)

        return [name.strip(), value.strip()]