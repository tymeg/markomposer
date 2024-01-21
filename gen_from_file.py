from gen_music import *


class FromFileMusicGenerator(MusicGenerator):
    '''Object representing the generator of music from nanoGPT's output.'''
    def __init__(self, mm: MarkovModel):
        '''Saves the MarkovModel.'''
        self.mm = mm

    # very similar to method 2 for now!
    def generate_music_from_file_nanogpt(
        self,
        input_filepath: str,
        output_file: str,
        instrument: int = 0,
        velocity: int = utils.DEFAULT_VELOCITY,
        tempo: Optional[int] = utils.DEFAULT_TEMPO,
        lengths_flatten_factor: Optional[int] = None,
        strict_time_signature: Optional[bool] = False,
        broad_chords: bool = False,
    ) -> None:
        '''
        Generates music, parsing tokens from nanoGPT in input_filepath.
        Flattens the lengths or moves note starts to strong beats, if specified.
        Saves the track to .mid output_file.
        '''
        new_mid = MidiFile(
            type=0, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT
        )  # one track
        track = super()._start_track(new_mid, instrument, True)
        super()._set_tempo(track, tempo)
        super()._set_key(track)

        in_time_signature = self.mm.fixed_time_signature
        bar_length = self.mm.main_bar_length
        if in_time_signature:
            beats_per_bar, _ = super()._set_time_signature(track)
            strong_beat_length = super()._calculate_strong_beat_length(
                bar_length, beats_per_bar
            )
            time_in_strong_beat = 0

        # with open(input_filepath) as f:
        #     tuples = f.read().split()
        with open(input_filepath) as f:
            values = f.read().split()

        messages = (
            list()
        )  # list of tuples (absolute start time, note, if_note_on, velocity, channel)
        # ADDING MESSAGES LOOP
        chord = set()
        total_time = 0
        # tuples.pop(0) # get rid of START
        # for tuple in tuples:
        #     next_note, note_length, until_next_note_start = map(int, tuple.split(","))
        values.pop(0)  # get rid of START
        while values:
            until_next_note_start, next_note, note_length = (
                int(values[0][1:]),
                int(values[1][1:]),
                int(values[2][1:]),
            )
            for _ in range(3):
                values.pop(0)

            if lengths_flatten_factor is not None or self.mm.fixed_time_signature:
                note_length, until_next_note_start = self.mm.round_time(
                    note_length,
                    True,
                    self.mm.fixed_time_signature,
                    lengths_flatten_factor,
                ), self.mm.round_time(
                    until_next_note_start,
                    False,
                    self.mm.fixed_time_signature,
                    lengths_flatten_factor,
                )

            if until_next_note_start == 0 and len(chord) == utils.MAX_CHORD_SIZE - 1:
                # start new chord
                until_next_note_start = utils.UNTIL_NEXT_CHORD * note_length

            offset = 0
            if in_time_signature:
                time_in_strong_beat = total_time % strong_beat_length
                time_in_bar = total_time % bar_length
                if (  # maybe extend until_next_note_start
                    time_in_strong_beat + note_length
                    > strong_beat_length
                ):
                    offset = (
                        strong_beat_length - time_in_strong_beat
                    ) % strong_beat_length
                    if strict_time_signature:
                        if time_in_bar + offset + note_length > bar_length:
                            offset = bar_length - time_in_bar
                    time_in_strong_beat = 0

                time_in_strong_beat += until_next_note_start
                if (  # maybe shrink until_next_note_start
                    time_in_strong_beat > strong_beat_length
                    and time_in_strong_beat % strong_beat_length != 0
                ):
                    until_next_note_start -= time_in_strong_beat - strong_beat_length

            total_time += offset
            if next_note not in chord and (not broad_chords or super().__good_note(next_note, chord)):
                messages.append((total_time, next_note, True, velocity, 0))
                messages.append(
                    (total_time + note_length, next_note, False, velocity, 0)
                )
            else:
                continue

            end_of_chord = False
            if until_next_note_start == 0:
                chord.add(next_note)
            else:
                if len(chord) >= 1:
                    end_of_chord = True
                    chord_size = len(chord) + 1
                chord = set()

            if in_time_signature and end_of_chord:
                # unifying chord's notes' starts
                start_messages = messages[-2 * chord_size :: 2]
                chord_start_time = max(start_messages, key=lambda m: m[0])[0]

                for msg_idx in range(len(messages) - 2 * chord_size, len(messages)):
                    start_time, note, note_on, velocity, channel = messages[msg_idx]
                    if msg_idx % 2 == 0:
                        messages[msg_idx] = (
                            chord_start_time,
                            note,
                            note_on,
                            velocity,
                            channel,
                        )
                        offset = chord_start_time - start_time
                    else:
                        messages[msg_idx] = (
                            start_time + offset,
                            note,
                            note_on,
                            velocity,
                            channel,
                        )
                total_time = chord_start_time
                until_next_note_start -= offset

            total_time += until_next_note_start

        super()._append_messages(track, messages)

        new_mid.save(os.path.join(os.getcwd(), output_file))


# pathname=None creates a MarkovModel object without processing any mids
mm = MarkovModel(
    n=None,
    dir=True,
    pathname=None,
    merge_tracks=None,
    ignore_bass=None,
    # lengths_flatten_factor=2,
)

# perform aligning notes to time signature's accents
# mm_4_4 = MarkovModel(
#     n=None,
#     dir=True,
#     pathname=None,
#     merge_tracks=None,
#     ignore_bass=None,
#     time_signature="4/4",
#     lengths_flatten_factor=2,
# )

generator = FromFileMusicGenerator(mm)
generator.generate_music_from_file_nanogpt(
    input_filepath="nanoGPT/test0.txt",
    output_file="test_gpt1.mid",
)
# generator.generate_music_from_file_nanogpt(
#     input_filepath="nanoGPT/test0.txt",
#     output_file="test_gpt2.mid",
#     lengths_flatten_factor=2,
#     tempo=80,
# )

# generator_4_4 = FromFileMusicGenerator(mm_4_4)
# generator_4_4.generate_music_from_file_nanogpt(
#     input_filepath="nanoGPT/test0.txt",
#     output_file="test_gpt4_4_4.mid",
#     tempo=80,
# )
