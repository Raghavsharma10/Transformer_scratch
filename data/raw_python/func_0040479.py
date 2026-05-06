def send_work(baseurl, work_id=None, filename=None, command="make"):
    """Ask user for a file to send to a work"""
    while 1:
        if not work_id:
            try:
                work_id = input("id? ")
            except KeyboardInterrupt:
                exit(0)
        work = get_work(work_id)
        if not work:
            print("id '{0}' not found".format(work_id))
            work_id = None
            continue
        if not work.is_open:  # Verify it is open
            print('"It\'s too late for {0} baby..." (Arnold Schwarzenegger)'.format(work.title))
            work_id = None
            continue
        if not filename:
            try:
                filename = input("filename? ")
            except KeyboardInterrupt:
                exit(0)
        while 1:
            try:
                if command:
                    if not archive_compile(filename, command):
                        print("Compilation failed")
                        try:
                            send = input("Send anyway [y/N] ")
                        except KeyboardInterrupt:
                            exit(0)
                        if send != "y":
                            exit(1)
                            return
                work.upload(baseurl, filename)
                print("Uplodaed, but should verify it on the website")
                return
            except FileNotFoundError:
                print("{0} not found in current dir".format(filename))
                filename = None