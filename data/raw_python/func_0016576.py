def main(source_samplerate, target_samplerate, params, converter_type):
    """Setup the resampling and audio output callbacks and start playback."""
    from time import sleep

    ratio = target_samplerate / source_samplerate

    with sr.CallbackResampler(get_input_callback(source_samplerate, params),
                              ratio, converter_type) as resampler, \
            sd.OutputStream(channels=1, samplerate=target_samplerate,
                            callback=get_playback_callback(
                                resampler, target_samplerate, params)):
        print("Playing back...  Ctrl+C to stop.")
        try:
            while True:
                sleep(1)
        except KeyboardInterrupt:
            print("Aborting.")