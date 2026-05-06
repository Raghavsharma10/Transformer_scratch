def broadcast(self, data_dict):
        '''
        Send to the visualizer (if there is one) or enqueue for later
        '''
        if self.vis_socket:
            self.queued_messages.append(data_dict)
            self.send_all_updates()