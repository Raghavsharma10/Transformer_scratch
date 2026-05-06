def main():
    """Ideally we shouldn't lose the first second of events"""
    time.sleep(1)
    with Input() as input_generator:
        for e in input_generator:
            print(repr(e))