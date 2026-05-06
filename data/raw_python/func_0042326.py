def piece_file(input_f, chunk_size):
        """
        Provides a streaming interface to file data in chunks of even size, which
        avoids memoryerrors from loading whole files into RAM to pass to `pieces`.
        """
        chunk = input_f.read(chunk_size)
        total_bytes = 0
        while chunk:
            yield chunk
            chunk = input_f.read(chunk_size)
            total_bytes += len(chunk)