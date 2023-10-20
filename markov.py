import os
import utils
from mido import MidiFile, MetaMessage, tempo2bpm
import math


class MarkovModel:
    def __init__(self, n: int, dir: bool, pathname: str, key: str = None) -> None:
        self.n = n  # n-grams

        # NOTES
        # n-grams
        self.note_ngrams = {}  # numbers -> how many
        self.note_ngrams_without_octaves = {}  # strings -> how many
        # n-1-grams
        self.note_nminus1grams = {}  # numbers -> how many
        self.note_nminus1grams_without_octaves = {}  # strings -> how many

        # default for beat=quarter note, changed in process_midi if beat value is different
        self.length_precision = utils.DEFAULT_LENGTH_PRECISION

        self.lengths_range = utils.DEFAULT_LENGTHS_RANGE

        # NOTE LENGTHS AND INTERVALS
        self.length_ngrams = {}
        self.length_nminus1grams = {}
        self.note_length_counts = {}
        self.interval_counts = {}

        # TUPLES (NOTE, NOTE LENGTH, TIME UNTIL NEXT NOTE START) - MELODY WITH HARMONY
        self.tuple_ngrams = {}
        self.tuple_ngrams_without_octaves = {}
        self.tuple_nminus1grams = {}
        self.tuple_nminus1grams_without_octaves = {}

        # TUPLES (NOTE, NOTE LENGTH, INTERVAL) - MELODY
        self.melody_ngrams = {}
        self.melody_ngrams_without_octaves = {}
        self.melody_nminus1grams = {}
        self.melody_nminus1grams_without_octaves = {}

        # main is currently first, maybe it should be the longest/most often among mid files?
        # if stays 0, will be set default later in music generation
        self.current_beats_per_bar, self.main_beats_per_bar = 0, 0
        self.current_beat_value, self.main_beat_value = 0, 0
        # but main_tempo is average tempo
        self.current_tempo, self.main_tempo, self.tempos_count = 0, 0, 0
        # given or None (don't force any specific key)
        self.main_key = key
        self.current_key = ""

        self.path = os.path.join(os.getcwd(), pathname)  # CWD
        # self.path = os.path.join(os.path.dirname(__file__), pathname) # directory of markov.py
        self.mids = []
        self.processed_mids = 0

        self.__collect_mid_files(dir)

        for mid in self.mids:
            self.__process_mid_file(mid)

        if self.tempos_count > 1:
            self.main_tempo = self.main_tempo // self.tempos_count  # average

    def __collect_mid_files(self, dir: bool) -> None:
        if dir:
            for filename in os.listdir(self.path):
                file = os.path.join(self.path, filename)
                if (
                    os.path.isfile(file)
                    and os.path.splitext(file)[-1].lower() == ".mid"
                ):
                    mid_file = MidiFile(file)
                    if not mid_file.type == 2:
                        self.mids.append(mid_file)
                    else:
                        print(f"Skipped {mid_file.filename} - type 2!")
            if not self.mids:
                raise ValueError("No .mid files of type 0 or 1 in given directory!")
        else:  # assumes file is of .mid extension
            mid_file = MidiFile(self.path)
            if mid_file.type == 2:
                raise ValueError(".mid file should be of type 0 or 1!")
            self.mids.append(mid_file)

    def __round_time(self, length: int, ticks_per_beat_factor: float, up: bool) -> int:
        round_fun = math.ceil if up else math.floor

        rounded_length = (
            round_fun(length * ticks_per_beat_factor / self.length_precision)
            * self.length_precision
        )
        if rounded_length > self.lengths_range:
            rounded_length = self.lengths_range
        return rounded_length

    def __process_mid_file(self, mid: MidiFile) -> None:
        print(f"Mid's name: {mid.filename}")
        print(f"Mid's length [sec]: {mid.length}")
        print(f"File type: {mid.type}")

        print(f"Ticks per beat: {mid.ticks_per_beat}")
        # to count lengths properly - TODO: but does it mess up tempo?
        ticks_per_beat_factor = utils.DEFAULT_TICKS_PER_BEAT / mid.ticks_per_beat
        # list of pairs: (start in total ticks, key)
        keys = []

        for track_idx, track in enumerate(mid.tracks):
            total_time = 0
            key_idx = 0

            # list of tuples (start of note, note length, note) to sort lexycographically
            note_lengths = []
            # times in ticks between end of previous (last) note and start of next
            intervals = []
            interval = 0
            currently_playing_notes_starts = {}

            print(f"Track {track_idx}: {track.name}")
            for msg in track:
                # print(msg)
                total_time += msg.time
                if keys:
                    if key_idx < len(keys) - 1:
                        if total_time >= keys[key_idx + 1][0]:
                            key_idx += 1
                    self.current_key = keys[key_idx][1]

                if interval != -1:
                    interval += msg.time

                if msg.is_meta:
                    self.__read_meta_message(msg, total_time, keys)
                else:
                    if msg.type == "note_on" and msg.velocity > 0:
                        currently_playing_notes_starts[msg.note] = total_time
                        if interval != -1:
                            intervals.append(interval)
                            interval = -1
                        # print(utils.get_note_name_with_octave(msg.note))
                        # print(total_time)
                    if (
                        msg.type == "note_on" and msg.velocity == 0
                    ) or msg.type == "note_off":
                        start = currently_playing_notes_starts[msg.note]
                        interval = 0
                        note = msg.note

                        # SKIPS THE FILE!
                        if self.main_key and not self.current_key:
                            return

                        if self.main_key and self.current_key:
                            note = utils.transpose(
                                note, self.current_key, self.main_key
                            )
                        note_lengths.append((start, total_time - start, note))

            if note_lengths:
                note_lengths.sort()
                notes = list(map(lambda tpl: tpl[2], note_lengths))
                # print(f"Track {track_idx} notes: \n{notes}")

                time_between_note_starts = []
                for idx in range(1, len(note_lengths)):
                    rounded_time = self.__round_time(
                        note_lengths[idx][0] - note_lengths[idx - 1][0],
                        ticks_per_beat_factor,
                        False,
                    )
                    time_between_note_starts.append(rounded_time)

                (
                    melody_notes,
                    melody_note_lengths,
                    melody_intervals,
                ) = self.__extract_melody(note_lengths, ticks_per_beat_factor)

                rounded_note_lengths = list(
                    map(
                        lambda tpl: self.__round_time(
                            tpl[1], ticks_per_beat_factor, True
                        ),
                        note_lengths,
                    )
                )
                rounded_intervals = list(
                    map(
                        lambda interval: self.__round_time(
                            interval, ticks_per_beat_factor, False
                        ),
                        intervals,
                    )
                )

                melody_tuples = list(
                    zip(melody_notes, melody_note_lengths, melody_intervals)
                )
                all_tuples = list(
                    zip(notes, rounded_note_lengths, time_between_note_starts)
                )
                length_pairs = list(zip(melody_note_lengths, melody_intervals))

                self.__count_track_note_ngrams(notes)

                self.__count_track_length_ngrams(length_pairs)

                self.__count_track_tuple_ngrams(melody_tuples, True)
                self.__count_track_tuple_ngrams(all_tuples, False)

                self.__count_track_length_occurences(rounded_note_lengths, True)
                self.__count_track_length_occurences(rounded_intervals, False)

        self.processed_mids += 1

        # print(f"\nNotes n-grams: \n{self.note_ngrams}\n")
        # print(f"Notes n-grams without octaves: \n{self.note_ngrams_without_octaves}\n")
        # print(f"\nNotes n-1-grams: \n{self.note_nminus1grams}\n")
        # print(
        #     f"Notes n-1-grams without octaves: \n{self.note_nminus1grams_without_octaves}\n"
        # )

        # print(f"Note length counts: \n{self.note_length_counts}")
        # print(f"Interval counts: \n{self.interval_counts}")

    def __read_meta_message(
        self, msg: MetaMessage, total_time: int, keys: list[str]
    ) -> None:
        if msg.type == "set_tempo":
            self.current_tempo = msg.tempo
            self.main_tempo += self.current_tempo
            self.tempos_count += 1
            # print(
            #     f"Tempo: {tempo2bpm(current_tempo)} BPM ({current_tempo} microseconds per quarter note)"
            # )
        if msg.type == "time_signature":
            self.current_beats_per_bar = msg.numerator
            if self.main_beats_per_bar == 0:
                self.main_beats_per_bar = self.current_beats_per_bar
            self.current_beat_value = msg.denominator
            if self.main_beat_value == 0:
                self.main_beat_value = self.current_beat_value
                # good?
                self.length_precision = utils.DEFAULT_TICKS_PER_BEAT // (
                    32 // self.main_beat_value
                )
                self.lengths_range = (
                    2 * utils.DEFAULT_TICKS_PER_BEAT * self.main_beat_value
                )
            print(
                f"Time signature: {self.current_beats_per_bar}/{self.current_beat_value}"
            )
        if msg.type == "key_signature":
            keys.append((total_time, msg.key))
            print(f"Key: {msg.key}")

    def __extract_melody(
        self, note_lengths: list[tuple[int]], ticks_per_beat_factor: int
    ) -> tuple[list[int]]:
        # assumes melody is the sequence of single notes with highest pitch of all playing

        end_time = (
            note_lengths[len(note_lengths) - 1][0]
            + note_lengths[len(note_lengths) - 1][1]
        )
        note_lengths_dict = {nl: True for nl in note_lengths}

        # in each box we have notes which play in that time frame of length_precision:
        # (note, start, note_length)
        boxes = [[] for i in range(end_time // self.length_precision + 1)]
        for start, note_length, note in note_lengths:
            for box_idx in range(
                start // self.length_precision,
                math.ceil((start + note_length) / self.length_precision),
            ):
                # weird, sometimes gave errors
                if box_idx < len(boxes):
                    boxes[box_idx].append((note, start, note_length))

        boxes = list(map(sorted, boxes))

        for box_idx in range(len(boxes)):
            if len(boxes[box_idx]) > 1:
                # highest note in box
                hnote, hstart, hnote_length = boxes[box_idx][len(boxes[box_idx]) - 1]
                while len(boxes[box_idx]) > 1:
                    note, start, note_length = boxes[box_idx][0]
                    if note_lengths_dict.get((start, note_length, note)):
                        tolerance = self.length_precision // 3
                        if (
                            (hstart > start and start + note_length - tolerance > hstart)
                            or (start > hstart and hstart + hnote_length - tolerance > start)
                        ):
                            del note_lengths_dict[(start, note_length, note)]
                    boxes[box_idx].pop(0)

        full_melody_note_lengths = sorted(list(note_lengths_dict.keys()))
        melody_notes = list(map(lambda tpl: tpl[2], full_melody_note_lengths))
        # round up - I don't want 0 length note lengths
        melody_note_lengths = list(
            map(
                lambda tpl: self.__round_time(tpl[1], ticks_per_beat_factor, True),
                full_melody_note_lengths,
            )
        )

        melody_intervals = []
        for idx in range(1, len(full_melody_note_lengths)):
            rounded_interval = (
                self.__round_time(
                    full_melody_note_lengths[idx][0]
                    - full_melody_note_lengths[idx - 1][0],
                    ticks_per_beat_factor,
                    False,
                )
                - melody_note_lengths[idx - 1]
            )
            if rounded_interval < 0:
                rounded_interval = 0

            melody_intervals.append(rounded_interval)
        melody_intervals.append(0)

        return melody_notes, melody_note_lengths, melody_intervals

    def __count_ngram(self, ngrams: dict[tuple], counted_ngram: tuple) -> None:
        if ngrams.get(counted_ngram) is not None:
            ngrams[counted_ngram] += 1
        else:
            ngrams[counted_ngram] = 1

    def __count_track_note_ngrams(self, notes: list[int]) -> None:
        # UGLY!
        for note_idx in range(len(notes) - self.n + 2):
            # count n-1-gram
            nminus1gram = tuple(
                [notes[note_idx + offset] for offset in range(self.n - 1)]
            )
            nminus1gram_without_octaves = tuple(
                [
                    utils.get_note_name(notes[note_idx + offset])
                    for offset in range(self.n - 1)
                ]
            )

            self.__count_ngram(self.note_nminus1grams, nminus1gram)
            self.__count_ngram(
                self.note_nminus1grams_without_octaves, nminus1gram_without_octaves
            )

            # not last n-1 notes -> count n-gram
            if note_idx != len(notes) - self.n + 1:
                ngram = nminus1gram + (notes[note_idx + self.n - 1],)
                ngram_without_octaves = nminus1gram_without_octaves + (
                    utils.get_note_name(notes[note_idx + self.n - 1]),
                )

                self.__count_ngram(self.note_ngrams, ngram)
                self.__count_ngram(
                    self.note_ngrams_without_octaves, ngram_without_octaves
                )

    def __count_track_tuple_ngrams(
        self, tuples: list[tuple[int]], melody: bool
    ) -> None:
        # ugly
        if melody:
            ngrams = self.melody_ngrams
            ngrams_without_octaves = self.melody_ngrams_without_octaves
            nminus1grams = self.melody_nminus1grams
            nminus1grams_without_octaves = self.melody_nminus1grams_without_octaves
        else:
            ngrams = self.tuple_ngrams
            ngrams_without_octaves = self.tuple_ngrams_without_octaves
            nminus1grams = self.tuple_nminus1grams
            nminus1grams_without_octaves = self.tuple_nminus1grams_without_octaves

        for note_idx in range(len(tuples) - self.n + 2):
            # count n-1-gram
            nminus1gram = tuple(
                [tuples[note_idx + offset] for offset in range(self.n - 1)]
            )
            nminus1gram_without_octaves = tuple(
                map(
                    lambda tpl: (utils.get_note_name(tpl[0]), tpl[1], tpl[2]),
                    nminus1gram,
                )
            )

            self.__count_ngram(nminus1grams, nminus1gram)
            self.__count_ngram(
                nminus1grams_without_octaves, nminus1gram_without_octaves
            )

            # not last n-1 tuples -> count n-gram
            if note_idx != len(tuples) - self.n + 1:
                ngram = nminus1gram + (tuples[note_idx + self.n - 1],)
                ngram_without_octaves = nminus1gram_without_octaves + (
                    (
                        utils.get_note_name(tuples[note_idx + self.n - 1][0]),
                        tuples[note_idx + self.n - 1][1],
                        tuples[note_idx + self.n - 1][2],
                    ),
                )

                self.__count_ngram(ngrams, ngram)
                self.__count_ngram(ngrams_without_octaves, ngram_without_octaves)

    def __count_track_length_ngrams(self, length_pairs: list[tuple[int]]) -> None:
        for length_idx in range(len(length_pairs) - self.n + 2):
            # count n-1-gram
            nminus1gram = tuple(
                [length_pairs[length_idx + offset] for offset in range(self.n - 1)]
            )

            self.__count_ngram(self.length_nminus1grams, nminus1gram)

            # not last n-1 notes -> count n-gram
            if length_idx != len(length_pairs) - self.n + 1:
                ngram = nminus1gram + (length_pairs[length_idx + self.n - 1],)

                self.__count_ngram(self.length_ngrams, ngram)

    def __count_track_length_occurences(
        self, lengths: list[int], note_lengths: bool
    ) -> None:
        max_length = max(lengths)

        if note_lengths:
            counts = self.note_length_counts
        else:
            counts = self.interval_counts

        for length in range(0, max_length + 1, self.length_precision):
            if counts.get(length) is None:
                counts[length] = lengths.count(length)
            else:
                counts[length] += lengths.count(length)
