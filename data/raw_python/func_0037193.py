def main():
    "Main program"
    generators = check_dependencies()
    args = docopt(__doc__, version='md2ebook 0.0.1-dev')
    commander = Commander(args, generators)
    commander.handle()