def mp_spawn(self):
        """ Spawn worker processes (using multiprocessing) """
        processes = []
        for x in range(self.queue_worker_amount):
            process = multiprocessing.Process(target=self.mp_worker)
            process.start()
            processes.append(process)
        for process in processes:
            process.join()