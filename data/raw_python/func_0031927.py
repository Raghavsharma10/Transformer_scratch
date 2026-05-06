def run(self):
        """Run the builder on changes"""
        event_handler = Handler()
        threads = []
        paths = [os.path.join(cwd, "content"), os.path.join(cwd, "templates")]

        for i in paths:
            targetPath = str(i)
            self.observer.schedule(event_handler, targetPath, recursive=True)
            threads.append(self.observer)

        self.observer.start()

        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("\nObserver stopped.")

        self.observer.join()