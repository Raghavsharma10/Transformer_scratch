def main():
    """
    Entry point
    """
    client_1 = MessageBot("verne", "Jules Verne")
    client_1.start()
    client_1.connect("127.0.0.1")

    client_2 = MessageBot("adams", "Douglas Adams")
    client_2.start()
    client_2.connect("127.0.0.1")

    herald_1 = Herald(client_1)
    herald_1.start()

    herald_2 = Herald(client_2)
    herald_2.start()

    handler = LogHandler()
    herald_1.register('/toto/*', handler)
    herald_2.register('/toto/*', handler)

    cmd = HeraldBot("bot", "Robotnik", herald_1)
    cmd.connect("127.0.0.1")

    cmd.wait_stop()

    for closable in (client_1, client_2, herald_1, herald_2):
        closable.close()

    logging.info("Bye !")