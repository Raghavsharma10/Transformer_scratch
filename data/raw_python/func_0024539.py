def clear_queue(self):
        """
        clear outs all messages from INPUT_QUEUE_NAME
        """
        def remove_message(ch, method, properties, body):
            print("Removed message: %s" % body)
        self.input_channel.basic_consume(remove_message, queue=self.INPUT_QUEUE_NAME, no_ack=True)
        try:
            self.input_channel.start_consuming()
        except (KeyboardInterrupt, SystemExit):
            log.info(" Exiting")
            self.exit()