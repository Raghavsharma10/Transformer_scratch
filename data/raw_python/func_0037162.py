def handle(self, *args, **options):
        """Run the managemement command."""
        if options['clean']:
            clean()

        if options['local']:
            local()

        if options['remote']:
            results = remote()
            render = lambda t: render_to_string(t, results)
            if options['notify']:
                send_mail(
                    options['subject'],
                    render('summary.txt'),
                    options['from'],
                    [options['notify']],
                    html_message=render('summary.html'),
                    fail_silently=False,
                )