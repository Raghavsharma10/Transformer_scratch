def check_packet(self):
        '''is there a valid packet (from another thread) for this app/instance?'''
        if not os.path.exists(self.packet_file()):
            # No packet file, we're good
            return True
        else:
            # There's already a file, but is it still running?
            try:
                with open(self.packet_file()) as f:
                    packet = json.loads(f.read())
                if time.time() - packet['last_time'] > 3.0*packet['poll_time']:
                    # We haven't heard a ping in too long. It's probably dead
                    return True
                else:
                    # Still getting pings.. probably still a live process
                    return False
            except:
                # Failed to read file... try again in a second
                time.sleep(random.random()*2)
                return self.check_packet()