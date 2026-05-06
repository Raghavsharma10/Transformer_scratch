def growl(text):
    """send native notifications where supported. Growl is gone."""
    if platform.system() == 'Darwin':
        import pync
        pync.Notifier.notify(text, title="Hitman")

    elif platform.system() == 'Linux':
        notified = False
        try:
            logger.debug("Trying to import pynotify")
            import pynotify
            pynotify.init("Hitman")
            n = pynotify.Notification("Hitman Status Report", text)
            n.set_timeout(pynotify.EXPIRES_DEFAULT)
            n.show()
            notified = True
        except ImportError:
            logger.debug("Trying notify-send")
            # print("trying to notify-send")
            if Popen(['which', 'notify-send'], stdout=PIPE).communicate()[0]:
                # Do an OSD-Notify
                # notify-send "Totem" "This is a superfluous notification"
                os.system("notify-send \"Hitman\" \"%r\" " % str(text))
                notified = True
        if not notified:
            try:
                logger.info("notificatons gnome gi???")
                import gi
                gi.require_version('Notify', '0.7')
                from gi.repository import Notify
                Notify.init("Hitman")
                # TODO have Icon as third argument.
                notification = Notify.Notification.new("Hitman", text)
                notification.show()
                Notify.uninit()
                notified = True
            except ImportError:
                logger.exception()
    elif platform.system() == 'Haiku':
        os.system("notify --type information --app Hitman --title 'Status Report' '%s'" % str(text))
    elif platform.system() == 'Windows':
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(text, "Hitman")
            # gntplib.publish("Hitman", "Status Update", "Hitman", text=text)
        except Exception:
            logger.exception()