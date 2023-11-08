import os
import utils
from mido import MidiFile, MetaMessage, tempo2bpm, bpm2tempo
import math
from typing import Dict


class MarkovModel:
    def __init__(
        self,
        n: int,
        dir: bool,
        pathname: str,
        merge_tracks: bool,
        ignore_bass: bool,
        key: str = None,
        tempo: int = None,
        extra_tempo_flatten: bool = False,
        time_signature: str = None,
    ) -> None:
        self.n = n  # n-grams

        # NOTES
        # n-grams
        self.note_ngrams = {}  # numbers -> how many
        self.note_ngrams_without_octaves = {}  # strings -> how many
        # n-1-grams
        self.note_nminus1grams = {}  # numbers -> how many
        self.note_nminus1grams_without_octaves = {}  # strings -> how many

        # NOTE LENGTHS AND INTERVALS
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
        self.__current_beats_per_bar, self.main_beats_per_bar = 0, utils.DEFAULT_BEATS_PER_BAR
        self.__current_beat_value, self.main_beat_value = 0, utils.DEFAULT_BEAT_VALUE
        if time_signature:
            self.main_beats_per_bar, self.main_beat_value = map(int, time_signature.split("/"))

        # but main_tempo is average tempo
        (
            self.__current_tempo,
            self.main_tempo,
            self.__tempos_count,
            self.__tempo_length,
        ) = (0, 0, 0, 0)
        self.fixed_tempo = False
        self.shortest_note = utils.SHORTEST_NOTE
        if tempo:
            self.fixed_tempo = True
            self.main_tempo = bpm2tempo(tempo)
            # tempo flattenization/normalization
            self.shortest_note = utils.SHORTEST_NOTE // 2
        if extra_tempo_flatten:
            self.shortest_note = utils.SHORTEST_NOTE // 4

        # default for beat=quarter note, changed in process_midi if beat value is different
        self.length_precision = utils.DEFAULT_TICKS_PER_BEAT // (
            self.shortest_note // self.main_beat_value
        )

        self.max_length = (
            utils.LONGEST_IN_BARS
            * self.main_beats_per_bar
            * utils.DEFAULT_TICKS_PER_BEAT
        )

        # given or None (don't force any specific key)
        self.main_key = key
        self.__current_key = None
        self.__keys = []

        self.path = os.path.join(os.getcwd(), pathname)  # CWD
        # self.path = os.path.join(os.path.dirname(__file__), pathname) # directory of markov.py
        self.notes_list_file1 = open(
            os.path.join(os.path.dirname(__file__), "nanoGPT/data/music/input1.txt"),
            "w",
        )
        self.notes_list_file2 = open(
            os.path.join(os.path.dirname(__file__), "nanoGPT/data/music/input2.txt"),
            "w",
        )

        self.mids = []
        self.processed_mids = 0

        self.__collect_mid_files(dir)

        for mid in self.mids:
            self.__process_mid_file(mid, merge_tracks, ignore_bass)

        if self.main_tempo > 0 and not self.fixed_tempo:
            self.main_tempo //= self.__tempos_count

        self.notes_list_file1.close()
        self.notes_list_file2.close()

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
        if rounded_length > self.max_length:
            rounded_length = self.max_length
        return rounded_length

    def __count_all(
        self, note_lengths: list[tuple[int]], ticks_per_beat_factor: int
    ) -> None:
        if note_lengths:
            note_lengths.sort()
            notes = list(map(lambda tpl: tpl[1], note_lengths))
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
                    lambda tpl: self.__round_time(tpl[2], ticks_per_beat_factor, True),
                    note_lengths,
                )
            )

            melody_tuples = list(
                zip(melody_notes, melody_note_lengths, melody_intervals)
            )
            all_tuples = list(
                zip(notes, rounded_note_lengths, time_between_note_starts)
            )

            # for generate_in_time_signature
            self.__count_track_note_ngrams(melody_notes)
            self.__count_track_length_occurences(
                melody_note_lengths, self.note_length_counts
            )
            self.__count_track_length_occurences(melody_intervals, self.interval_counts)

            # for generate_in_melody/tuple_ngrams
            self.__count_track_tuple_ngrams(
                melody_tuples,
                self.melody_ngrams,
                self.melody_ngrams_without_octaves,
                self.melody_nminus1grams,
                self.melody_nminus1grams_without_octaves,
            )
            self.__count_track_tuple_ngrams(
                all_tuples,
                self.tuple_ngrams,
                self.tuple_ngrams_without_octaves,
                self.tuple_nminus1grams,
                self.tuple_nminus1grams_without_octaves,
            )

            # append to file for nanoGPT
            for note, note_length, until_next_note_start in all_tuples:
                self.notes_list_file1.write(
                    f"N{str(note)} L{str(note_length)} I{str(until_next_note_start)} "
                )
                self.notes_list_file2.write(
                    f"{str(note)},{str(note_length)},{str(until_next_note_start)} "
                )

    def __transpose_track(self, note_lengths: list[tuple[int]]) -> list[tuple[int]]:
        notes_str = list(map(lambda tpl: utils.get_note_name(tpl[1]), note_lengths))
        self.__current_key = utils.infer_key(notes_str)
        if self.__current_key is None:
            return None

        print(f"Inferred key: {self.__current_key}")
        if self.main_key != self.__current_key:
            note_lengths = list(
                map(
                    lambda tpl: (
                        tpl[0],
                        utils.transpose(tpl[1], self.__current_key, self.main_key),
                        tpl[2],
                    ),
                    note_lengths,
                )
            )
        return note_lengths

    def __process_mid_file(
        self, mid: MidiFile, merge_tracks: bool, ignore_bass: bool
    ) -> int:
        print(f"Mid's name: {mid.filename}")
        print(f"Mid's length [sec]: {mid.length}")
        print(f"File type: {mid.type}")

        print(f"Ticks per beat: {mid.ticks_per_beat}")
        # to count lengths properly
        ticks_per_beat_factor = utils.DEFAULT_TICKS_PER_BEAT / mid.ticks_per_beat
        # list of pairs: (start in total ticks, key)
        self.__keys = []
        # dict: channel -> instrument
        instruments = {ch: 0 for ch in range(16)}

        if merge_tracks:
            all_note_lengths = []
            merged_tracks = 0

        if self.main_key:
            first_notes_track = True

        for track_idx, track in enumerate(mid.tracks):
            total_time = 0
            key_idx = 0

            # list of tuples (start of note, note length, note) to sort lexycographically
            note_lengths = []
            currently_playing_notes_starts = {}

            print(f"Track {track_idx}: {track.name}")
            for msg in track:
                # print(msg)
                total_time += msg.time
                self.__tempo_length += msg.time
                if self.__keys:
                    if key_idx < len(self.__keys) - 1:
                        if total_time >= self.__keys[key_idx + 1][0]:
                            key_idx += 1
                    self.__current_key = self.__keys[key_idx][1]

                if msg.is_meta:
                    self.__read_meta_message(msg, total_time)
                elif (  # ignore drums and percussive instruments/sound effects
                    msg.channel != utils.DRUM_CHANNEL
                    and instruments[msg.channel] < utils.PERCUSSIVE_INSTRUMENTS
                ):
                    if msg.type == "note_on" and msg.velocity > 0:
                        if (
                            not ignore_bass
                            or instruments[msg.channel] not in utils.BASS
                        ):
                            currently_playing_notes_starts[msg.note] = total_time
                            # print(utils.get_note_name_with_octave(msg.note))
                            # print(total_time)
                    elif (
                        msg.type == "note_on" and msg.velocity == 0
                    ) or msg.type == "note_off":
                        if (
                            not ignore_bass
                            or instruments[msg.channel] not in utils.BASS
                        ):
                            if currently_playing_notes_starts.get(msg.note):
                                start = currently_playing_notes_starts[msg.note]
                                note = msg.note

                                if (
                                    self.main_key
                                    and self.__current_key
                                    and self.main_key != self.__current_key
                                ):
                                    note = utils.transpose(
                                        note, self.__current_key, self.main_key
                                    )
                                note_lengths.append((start, note, total_time - start))
                    elif msg.type == "program_change":
                        instruments[msg.channel] = msg.program
                        print(msg.program)

            if note_lengths:
                if merge_tracks:
                    all_note_lengths += note_lengths
                    merged_tracks += 1
                    if merged_tracks == utils.MAX_TRACKS_TO_MERGE:
                        break  # ignore the rest or count them individually?
                else:
                    if (
                        self.main_key and not self.__current_key and first_notes_track
                    ):  # only happens on first track with notes
                        note_lengths = self.__transpose_track(note_lengths)
                        if note_lengths is None:
                            return  # skip file in training
                        first_notes_track = False
                    self.__count_all(note_lengths, ticks_per_beat_factor)

        if merge_tracks:
            if self.main_key and not self.__current_key and note_lengths:
                all_note_lengths = self.__transpose_track(all_note_lengths)
                if all_note_lengths is None:
                    return  # skip file in training
            self.__count_all(all_note_lengths, ticks_per_beat_factor)

        self.main_tempo += self.__current_tempo
        self.__tempos_count += 1
        self.__current_key = None
        self.processed_mids += 1

        # print(f"\nNotes n-grams: \n{self.note_ngrams}\n")
        # print(f"Notes n-grams without octaves: \n{self.note_ngrams_without_octaves}\n")
        # print(f"\nNotes n-1-grams: \n{self.note_nminus1grams}\n")
        # print(
        #     f"Notes n-1-grams without octaves: \n{self.note_nminus1grams_without_octaves}\n"
        # )

        # print(f"Note length counts: \n{self.note_length_counts}")
        # print(f"Interval counts: \n{self.interval_counts}")

    def __read_meta_message(self, msg: MetaMessage, total_time: int) -> None:
        if msg.type == "set_tempo" and not self.fixed_tempo:
            if self.__current_tempo:
                self.main_tempo += self.__current_tempo
                self.__tempos_count += 1
            self.__current_tempo = msg.tempo
            # print(
            #     f"Tempo: {tempo2bpm(self.__current_tempo)} BPM ({self.__current_tempo} microseconds per quarter note)"
            # )
        if msg.type == "time_signature":
            self.__current_beats_per_bar = msg.numerator
            self.__current_beat_value = msg.denominator
            # if self.main_beats_per_bar == 0 and self.main_beat_value == 0:
                # self.main_beats_per_bar = self.__current_beats_per_bar
                # self.main_beat_value = self.__current_beat_value
                # good?
                # self.length_precision = utils.DEFAULT_TICKS_PER_BEAT // (
                #     utils.SHORTEST_NOTE // self.main_beat_value
                # )
                # self.max_length = (
                #     utils.LONGEST_IN_BARS
                #     * self.main_beats_per_bar
                #     * utils.DEFAULT_TICKS_PER_BEAT
                # )
            print(
                f"Time signature: {self.__current_beats_per_bar}/{self.__current_beat_value}"
            )
        if msg.type == "key_signature":
            self.__keys.append((total_time, msg.key))
            print(f"Key: {msg.key}")

    def __extract_melody(
        self, note_lengths: list[tuple[int]], ticks_per_beat_factor: int
    ) -> tuple[list[int]]:
        # assumes melody is the sequence of single notes with highest pitch of all playing

        end_time = (
            note_lengths[len(note_lengths) - 1][0]
            + note_lengths[len(note_lengths) - 1][2]
        )
        note_lengths_dict = {nl: True for nl in note_lengths}

        # in each box we have notes which play in that time frame of length_precision:
        # (note, start, note_length)
        boxes = [[] for i in range(end_time // self.length_precision + 1)]
        for start, note, note_length in note_lengths:
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
                _, hstart, hnote_length = boxes[box_idx][len(boxes[box_idx]) - 1]
                while len(boxes[box_idx]) > 1:
                    note, start, note_length = boxes[box_idx][0]
                    if note_lengths_dict.get((start, note, note_length)):
                        tolerance = self.length_precision // 3
                        if (
                            hstart > start and start + note_length - tolerance > hstart
                        ) or (
                            start > hstart and hstart + hnote_length - tolerance > start
                        ):
                            del note_lengths_dict[(start, note, note_length)]
                    boxes[box_idx].pop(0)

        full_melody_note_lengths = sorted(list(note_lengths_dict.keys()))
        melody_notes = list(map(lambda tpl: tpl[1], full_melody_note_lengths))
        # round up - I don't want 0 length note lengths
        melody_note_lengths = list(
            map(
                lambda tpl: self.__round_time(tpl[2], ticks_per_beat_factor, True),
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

    def __count_ngram(self, ngrams: Dict[tuple, int], counted_ngram: tuple) -> None:
        if ngrams.get(counted_ngram) is not None:
            ngrams[counted_ngram] += 1
        else:
            ngrams[counted_ngram] = 1

    def __count_track_note_ngrams(self, notes: list[int]) -> None:
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
        self,
        tuples: list[tuple[int]],
        ngrams: list[tuple[int]],
        ngrams_without_octaves: list[tuple[str]],
        nminus1grams: list[tuple[int]],
        nminus1grams_without_octaves: list[tuple[str]],
    ) -> None:
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

    def __count_track_length_occurences(
        self, lengths: list[int], counts: list[int]
    ) -> None:
        max_length = max(lengths)

        for length in range(0, max_length + 1, self.length_precision):
            if counts.get(length) is None:
                counts[length] = lengths.count(length)
            else:
                counts[length] += lengths.count(length)
