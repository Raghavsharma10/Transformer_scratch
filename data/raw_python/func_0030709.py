def main():
    """
    Run me with the filename of a benchmark script as an argument.  I will time
    it and append the results to a file named output in the current working
    directory.
    """
    name = sys.argv[1]
    path = filepath.FilePath('.stat').temporarySibling()
    path.makedirs()
    func = makeBenchmarkRunner(path, sys.argv[1:])
    try:
        bench(name, path, func)
    finally:
        path.remove()