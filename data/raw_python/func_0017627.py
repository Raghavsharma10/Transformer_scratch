def handle_events(self):
        """
        An event handler that processes events from stdin and calls the on_click
        function of the respective object. This function is run in another
        thread, so as to not stall the main thread.
        """
        for event in sys.stdin:
            if event.startswith('['):
                continue
            name = json.loads(event.lstrip(','))['name']
            for obj in self.loader.objects:
                if obj.output_options['name'] == name:
                    obj.on_click(json.loads(event.lstrip(',')))