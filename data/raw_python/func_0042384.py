def main():
    """Main method for running upsidedown.py from the command line."""
    import sys

    output = []
    line = sys.stdin.readline()

    while line:
        line = line.strip("\n")
        output.append(transform(line))

        line = sys.stdin.readline()
    output.reverse()
    print("\n".join(output))