def get_environment(self):
        """
        Get environment facts.

        power and fan are currently not implemented
        cpu is using 1-minute average
        cpu hard-coded to cpu0 (i.e. only a single CPU)
        """
        environment = {}
        cpu_cmd = 'show proc cpu'
        mem_cmd = 'show memory statistics'
        temp_cmd = 'show env temperature status'

        output = self._send_command(cpu_cmd)
        environment.setdefault('cpu', {})
        environment['cpu'][0] = {}
        environment['cpu'][0]['%usage'] = 0.0
        for line in output.splitlines():
            if 'CPU utilization' in line:
                # CPU utilization for five seconds: 2%/0%; one minute: 2%; five minutes: 1%
                cpu_regex = r'^.*one minute: (\d+)%; five.*$'
                match = re.search(cpu_regex, line)
                environment['cpu'][0]['%usage'] = float(match.group(1))
                break

        output = self._send_command(mem_cmd)
        for line in output.splitlines():
            if 'Processor' in line:
                _, _, _, proc_used_mem, proc_free_mem = line.split()[:5]
            elif 'I/O' in line or 'io' in line:
                _, _, _, io_used_mem, io_free_mem = line.split()[:5]
        used_mem = int(proc_used_mem) + int(io_used_mem)
        free_mem = int(proc_free_mem) + int(io_free_mem)
        environment.setdefault('memory', {})
        environment['memory']['used_ram'] = used_mem
        environment['memory']['available_ram'] = free_mem

        environment.setdefault('temperature', {})
        # The 'show env temperature status' is not ubiquitous in Cisco IOS
        output = self._send_command(temp_cmd)
        if '% Invalid' not in output:
            for line in output.splitlines():
                if 'System Temperature Value' in line:
                    system_temp = float(line.split(':')[1].split()[0])
                elif 'Yellow Threshold' in line:
                    system_temp_alert = float(line.split(':')[1].split()[0])
                elif 'Red Threshold' in line:
                    system_temp_crit = float(line.split(':')[1].split()[0])
            env_value = {'is_alert': system_temp >= system_temp_alert,
                         'is_critical': system_temp >= system_temp_crit, 'temperature': system_temp}
            environment['temperature']['system'] = env_value
        else:
            env_value = {'is_alert': False, 'is_critical': False, 'temperature': -1.0}
            environment['temperature']['invalid'] = env_value

        # Initialize 'power' and 'fan' to default values (not implemented)
        environment.setdefault('power', {})
        environment['power']['invalid'] = {'status': True, 'output': -1.0, 'capacity': -1.0}
        environment.setdefault('fans', {})
        environment['fans']['invalid'] = {'status': True}

        return environment