import os
from mido import MidiFile, MetaMessage, tempo2bpm
from tqdm import tqdm
import math
from typing import List, Tuple, Dict

import utils


class MarkovModel:
    def __init__(
        self,
        n: int,
        dir: bool,
        pathname: str,
        merge_tracks: bool,
        ignore_bass: bool,
        key: str = None,
        time_signature: str = None,
        lengths_flatten_factor: int = None,
        allow_major_minor_transpositions: bool = False,
    ) -> None:
        self.n = n  # n-grams

        # TUPLES (NOTE, NOTE LENGTH, TIME FROM START TO START OF NEXT) - MELODY WITH HARMONY
        self.harmony_ngrams = dict()
        self.harmony_nminus1grams = dict()
        # (NOTE, NOTE LENGTH, TIME FROM START TO START OF NEXT, IF NOTE PITCH GOES UP)
        self.harmony_ngrams_without_octaves = dict()
        self.harmony_nminus1grams_without_octaves = dict()

        self.harmony_nminus1gram_starts_of_bar = set()
        self.harmony_nminus1gram_without_octaves_starts_of_bar = set()

        # TUPLES (NOTE, NOTE LENGTH, TIME FROM END TO START OF NEXT) - MELODY
        self.melody_ngrams = dict()
        self.melody_nminus1grams = dict()
        # (NOTE, NOTE LENGTH, TIME FROM END TO START OF NEXT, IF NOTE PITCH GOES UP)
        self.melody_ngrams_without_octaves = dict()
        self.melody_nminus1grams_without_octaves = dict()

        # nminus1grams which started some bar in input tracks
        self.melody_nminus1gram_starts_of_bar = set()
        self.melody_nminus1gram_without_octaves_starts_of_bar = set()

        # dict: chord -> how many
        self.chords = dict()
        self.chords_without_octaves = dict()

        # TUPLES (NOTE, NOTE LENGTH, TIME IN BAR)
        self.bar_ngrams = dict()
        self.bar_nminus1grams = dict()
        # (NOTE, NOTE LENGTH, TIME IN BAR, IF NOTE PITCH GOES UP)
        self.bar_ngrams_without_octaves = dict()
        self.bar_nminus1grams_without_octaves = dict()

        self.fixed_time_signature = False
        self.__current_beats_per_bar, self.__current_beat_value = 0, 0
        self.main_beats_per_bar, self.main_beat_value = (
            utils.DEFAULT_BEATS_PER_BAR,
            utils.DEFAULT_BEAT_VALUE,
        )
        if time_signature is not None:
            self.fixed_time_signature = True
            self.main_beats_per_bar, self.main_beat_value = map(
                int, time_signature.split("/")
            )

        # main_tempo is average tempo
        (
            self.__current_tempo,
            self.main_tempo,
            self.__tempos_count,
        ) = (0, 0, 0)

        self.__shortest_note = utils.SHORTEST_NOTE
        if lengths_flatten_factor is not None:
            self.__shortest_note //= lengths_flatten_factor

        # default: 32nd note
        self.length_precision = int(
            utils.DEFAULT_TICKS_PER_BEAT
            / (
                self.__shortest_note
                / utils.DEFAULT_BEAT_VALUE
                # self.__shortest_note // self.main_beat_value # ??
            )
        )

        self.main_bar_length = int(
            self.main_beats_per_bar
            * (utils.DEFAULT_TICKS_PER_BEAT / (self.main_beat_value / 4))
        )
        self.__bar_lengths = list()
        self.__current_bar_length = None

        # whole note or bar length, if it's shorter than whole note
        self.max_length = min(
            utils.DEFAULT_TICKS_PER_BEAT * 4,
            self.main_bar_length,
        )

        if self.main_beats_per_bar in [2, 3, 4]:
            self.simple_time = True
            self.used_note_lengths = list(
                map(
                    lambda l: utils.TICKS_PER_32NOTE * l,
                    utils.NOTE_LENGTHS_SIMPLE_TIME,
                )
            )
        else:
            self.simple_time = False
            self.used_note_lengths = list(
                map(
                    lambda l: utils.TICKS_PER_32NOTE * l,
                    utils.NOTE_LENGTHS_COMPOUND_TIME,
                )
            )

        # given or None (don't force any specific key)
        self.main_key = key
        self.__current_key = None
        self.__keys = list()
        self.__allow_major_minor_transpositons = allow_major_minor_transpositions

        if pathname is not None:
            self.path = os.path.join(os.getcwd(), pathname)  # CWD
            # self.path = os.path.join(os.path.dirname(__file__), pathname) # directory of markov.py
            self.notes_list_file1 = open(
                os.path.join(
                    os.path.dirname(__file__), "nanoGPT/data/music/input1.txt"
                ),
                "w",
            )
            self.notes_list_file2 = open(
                os.path.join(
                    os.path.dirname(__file__), "nanoGPT/data/music/input2.txt"
                ),
                "w",
            )

            self.mids = list()
            self.processed_mids = 0

            self.__collect_mid_files(dir)

            print("Processing mid tracks and creating Markov model...")
            for mid in tqdm(self.mids):
                self.notes_list_file1.write("START ")
                self.notes_list_file2.write("START ")
                self.__process_mid_file(mid, merge_tracks, ignore_bass)
                self.notes_list_file1.write("END\n")
                self.notes_list_file2.write("END\n")

            if self.main_tempo > 0:
                self.main_tempo //= self.__tempos_count
            else:
                self.main_tempo = utils.DEFAULT_TEMPO

            self.notes_list_file1.close()
            self.notes_list_file2.close()

            # print(dict(sorted(self.chords.items(), key=lambda item: item[1], reverse=True)))
            # print(dict(sorted(self.chords_without_octaves.items(), key=lambda item: item[1], reverse=True)))

    def __collect_mid_files(self, dir: bool) -> None:
        print("Collecting and parsing mid files...")
        if dir:
            for filename in tqdm(os.listdir(self.path)):
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

    def round_time(
        self,
        length: int,
        up: bool,
        in_time_signature: bool,
        lengths_flatten_factor: int = None,
    ) -> int:
        round_fun = math.ceil if up else math.floor
        length_precision = self.length_precision
        if lengths_flatten_factor:
            length_precision *= lengths_flatten_factor
        rounded_length = round_fun(length / length_precision) * length_precision

        if in_time_signature:
            for len_idx in range(1, len(self.used_note_lengths)):
                if (
                    self.used_note_lengths[len_idx - 1]
                    < length
                    <= self.used_note_lengths[len_idx]
                ):
                    rounded_length = self.used_note_lengths[len_idx]
                    break

        if rounded_length > self.max_length:
            rounded_length = self.max_length
        return rounded_length

    def __count_all(self, note_lengths: List[Tuple[int, bool]]) -> None:
        if note_lengths:
            note_lengths = list(set(note_lengths))
            note_lengths.sort()
            notes = list(map(lambda tpl: tpl[2], note_lengths))
            # print(f"Track {track_idx} notes: \n{notes}")

            time_between_note_starts = list()
            for idx in range(1, len(note_lengths)):
                rounded_time = self.round_time(
                    note_lengths[idx][0] - note_lengths[idx - 1][0],
                    False,
                    self.__shortest_note < utils.SHORTEST_NOTE,
                )
                time_between_note_starts.append(rounded_time)

            (
                melody_notes,
                melody_note_lengths,
                melody_intervals,
                melody_starts_of_bar,
            ) = self.__extract_melody_and_chords(note_lengths)

            rounded_note_lengths = list(
                map(
                    lambda tpl: self.round_time(
                        tpl[1], True, self.__shortest_note < utils.SHORTEST_NOTE
                    ),
                    note_lengths,
                )
            )
            starts_of_bar = list(map(lambda tpl: tpl[3], note_lengths))

            # always floor
            rounded_times_in_bar = list(
                map(
                    lambda tpl: (
                        math.floor(tpl[4] / self.length_precision)
                        * self.length_precision
                    )
                    % self.main_bar_length,
                    note_lengths,
                )
            )

            melody_tuples = list(
                zip(melody_notes, melody_note_lengths, melody_intervals)
            )
            all_tuples = list(
                zip(notes, rounded_note_lengths, time_between_note_starts)
            )
            bar_tuples = list(zip(notes, rounded_note_lengths, rounded_times_in_bar))

            # for generate_with_melody_ngrams
            self.__count_track_tuple_ngrams(
                melody_tuples,
                self.melody_ngrams,
                self.melody_ngrams_without_octaves,
                self.melody_nminus1grams,
                self.melody_nminus1grams_without_octaves,
                melody_starts_of_bar,
                self.melody_nminus1gram_starts_of_bar,
                self.melody_nminus1gram_without_octaves_starts_of_bar,
            )
            # for generate_with_tuple_ngrams
            self.__count_track_tuple_ngrams(
                all_tuples,
                self.harmony_ngrams,
                self.harmony_ngrams_without_octaves,
                self.harmony_nminus1grams,
                self.harmony_nminus1grams_without_octaves,
                starts_of_bar,
                self.harmony_nminus1gram_starts_of_bar,
                self.harmony_nminus1gram_without_octaves_starts_of_bar,
            )
            # for generate_with_bar_ngrams
            self.__count_track_tuple_ngrams(
                bar_tuples,
                self.bar_ngrams,
                self.bar_ngrams_without_octaves,
                self.bar_nminus1grams,
                self.bar_nminus1grams_without_octaves,
            )

            # append to file for nanoGPT
            for note, note_length, until_next_note_start in all_tuples:
                self.notes_list_file1.write(
                    f"I{str(until_next_note_start)} N{str(note)} L{str(note_length)} "
                )
                self.notes_list_file2.write(
                    f"{str(note)},{str(note_length)},{str(until_next_note_start)} "
                )

    def __transpose_track(
        self, note_lengths: List[Tuple[int, bool]]
    ) -> List[Tuple[int]]:
        notes_str = list(map(lambda tpl: utils.get_note_name(tpl[2]), note_lengths))
        self.__current_key = utils.infer_key(notes_str)
        if self.__current_key is None:
            return None

        # print(f"Inferred key: {self.__current_key}")
        if self.main_key != self.__current_key:
            note_lengths = list(
                map(
                    lambda tpl: (
                        tpl[0],
                        tpl[1],
                        utils.transpose(
                            tpl[2],
                            self.__current_key,
                            self.main_key,
                            self.__allow_major_minor_transpositons,
                        ),
                        tpl[3],
                        tpl[4],
                    ),
                    note_lengths,
                )
            )
        return note_lengths

    def __process_mid_file(
        self, mid: MidiFile, merge_tracks: bool, ignore_bass: bool
    ) -> int:
        # print(f"Mid's name: {mid.filename}")
        # print(f"Mid's length [sec]: {mid.length}")
        # print(f"File type: {mid.type}")

        # print(f"Ticks per beat: {mid.ticks_per_beat}")
        # to count lengths properly
        ticks_per_beat_factor = utils.DEFAULT_TICKS_PER_BEAT / mid.ticks_per_beat
        # list of pairs: (start in total ticks, key)
        self.__keys = list()
        # list of pairs: (start in total ticks, bar length)
        self.__bar_lengths = list()
        # dict: channel -> instrument
        instruments = {ch: 0 for ch in range(16)}

        if merge_tracks:
            all_note_lengths = list()
            merged_tracks = 0

        if self.main_key:
            first_notes_track = True

        for track_idx, track in enumerate(mid.tracks):
            total_time = 0
            key_idx = 0
            meter_idx = 0

            # list of tuples (start of note, note length, note, if start of bar, time in bar) to sort lexycographically
            note_lengths = list()
            currently_playing_notes_starts = dict()

            # print(f"Track {track_idx}: {track.name}")
            for msg in track:
                # print(msg)
                total_time += msg.time
                if self.__keys:
                    if key_idx < len(self.__keys) - 1:
                        if total_time >= self.__keys[key_idx + 1][0]:
                            key_idx += 1
                    self.__current_key = self.__keys[key_idx][1]
                if self.__bar_lengths:
                    if meter_idx < len(self.__bar_lengths) - 1:
                        if total_time >= self.__bar_lengths[meter_idx + 1][0]:
                            meter_idx += 1
                    self.__current_bar_length = self.__bar_lengths[meter_idx][1]

                if msg.is_meta:
                    self.__read_meta_message(msg, total_time)
                elif msg.type == "note_on" or msg.type == "note_off":
                    if (  # always ignore drums and percussive instruments/sound effects
                        msg.channel != utils.DRUM_CHANNEL
                        and instruments[msg.channel] < utils.PERCUSSIVE_INSTRUMENTS
                        and (
                            not ignore_bass
                            or instruments[msg.channel] not in utils.BASS
                        )
                    ):
                        if msg.type == "note_on" and msg.velocity > 0:
                            currently_playing_notes_starts[msg.note] = total_time
                        elif (
                            msg.type == "note_on" and msg.velocity == 0
                        ) or msg.type == "note_off":
                            if currently_playing_notes_starts.get(msg.note):
                                start = currently_playing_notes_starts[msg.note]
                                note = msg.note

                                time_in_bar = (
                                    int(
                                        (start - self.__bar_lengths[meter_idx][0])
                                        * ticks_per_beat_factor
                                    )
                                    % self.__current_bar_length
                                )
                                # determine if note starts bar
                                start_of_bar = False
                                if (
                                    time_in_bar < self.length_precision // 3
                                    or (self.__current_bar_length - time_in_bar)
                                    < self.length_precision // 3
                                ):  # arbitrary precision
                                    start_of_bar = True

                                if (
                                    self.main_key
                                    and self.__current_key
                                    and self.main_key != self.__current_key
                                ):
                                    note = utils.transpose(
                                        note,
                                        self.__current_key,
                                        self.main_key,
                                        self.__allow_major_minor_transpositons,
                                    )
                                note_lengths.append(
                                    (
                                        int(start * ticks_per_beat_factor),
                                        int(
                                            (total_time - start) * ticks_per_beat_factor
                                        ),
                                        note,
                                        start_of_bar,
                                        time_in_bar,
                                    )
                                )
                elif msg.type == "program_change":
                    instruments[msg.channel] = msg.program
                    # print(msg.program)

            if note_lengths:
                if merge_tracks:
                    all_note_lengths += note_lengths
                    merged_tracks += 1
                    # if merged_tracks == utils.MAX_TRACKS_TO_MERGE:
                    #     break  # ignore the rest or count them individually?
                else:
                    if (
                        self.main_key and not self.__current_key and first_notes_track
                    ):  # only happens on first track with notes
                        note_lengths = self.__transpose_track(note_lengths)
                        if note_lengths is None:
                            return  # skip file in training
                        first_notes_track = False
                    self.__count_all(note_lengths)

        if merge_tracks:
            if self.main_key and not self.__current_key and all_note_lengths:
                all_note_lengths = self.__transpose_track(all_note_lengths)
                if all_note_lengths is None:
                    return  # skip file in training
            self.__count_all(all_note_lengths)

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
        if msg.type == "set_tempo":
            if self.__current_tempo:
                self.main_tempo += self.__current_tempo
                self.__tempos_count += 1
            self.__current_tempo = msg.tempo
            # print(
            #     f"Tempo: {tempo2bpm(self.__current_tempo)} BPM ({self.__current_tempo} microseconds per quarter note)"
            # )
        elif msg.type == "time_signature":
            self.__current_beats_per_bar = msg.numerator
            self.__current_beat_value = msg.denominator
            self.__bar_lengths.append(
                (
                    total_time,
                    int(
                        self.__current_beats_per_bar
                        * (
                            utils.DEFAULT_TICKS_PER_BEAT
                            / (self.__current_beat_value / 4)
                        )
                    ),
                )
            )
            # print(
            #     f"Time signature: {self.__current_beats_per_bar}/{self.__current_beat_value}"
            # )
        elif msg.type == "key_signature":
            self.__keys.append((total_time, msg.key))
            # print(f"Key: {msg.key}")
        # else:
        # print(msg)

    def __extract_melody_and_chords(
        self, note_lengths: List[Tuple[int, bool]]
    ) -> Tuple[List[int], bool]:
        # assumes melody is the sequence of single notes with highest pitch of all playing

        end_time = note_lengths[-1][0] + note_lengths[-1][1]
        note_lengths_dict = {nl: True for nl in note_lengths}

        # in each box we have notes which play in that time frame of length_precision:
        # (note, start, note_length)
        boxes = [list() for _ in range(end_time // self.length_precision + 1)]
        for start, note_length, note, start_of_bar, time_in_bar in note_lengths:
            for box_idx in range(
                start // self.length_precision,
                math.ceil((start + note_length) / self.length_precision),
            ):
                # weird, sometimes gave errors
                if box_idx < len(boxes):
                    boxes[box_idx].append(
                        (note, start, note_length, start_of_bar, time_in_bar)
                    )

        boxes = list(map(sorted, boxes))

        for box_idx in range(len(boxes)):
            if len(boxes[box_idx]) > 1:
                # highest note in box
                hnote, hstart, hnote_length, _, _ = boxes[box_idx][
                    len(boxes[box_idx]) - 1
                ]
                extended_chord = (hnote,)
                chord = tuple()
                while len(boxes[box_idx]) > 1:
                    note, start, note_length, start_of_bar, time_in_bar = boxes[
                        box_idx
                    ][0]
                    if note_lengths_dict.get(
                        (start, note_length, note, start_of_bar, time_in_bar)
                    ):
                        chord += (note,)
                        extended_chord += (note,)
                        tolerance = self.length_precision // 3
                        if (
                            hstart > start and start + note_length - tolerance > hstart
                        ) or (
                            start > hstart and hstart + hnote_length - tolerance > start
                        ):
                            del note_lengths_dict[
                                (start, note_length, note, start_of_bar, time_in_bar)
                            ]
                    boxes[box_idx].pop(0)
                if len(extended_chord) > 2:
                    extended_chord = tuple(sorted(set(extended_chord)))
                    self.__count(self.chords, extended_chord)
                    self.__count(
                        self.chords_without_octaves,
                        tuple(
                            map(lambda note: utils.get_note_name(note), extended_chord)
                        ),
                    )
                if len(chord) > 1:
                    chord = tuple(sorted(set(chord)))
                    self.__count(self.chords, chord)
                    self.__count(
                        self.chords_without_octaves,
                        tuple(map(lambda note: utils.get_note_name(note), chord)),
                    )

        full_melody_note_lengths = sorted(list(note_lengths_dict.keys()))
        melody_notes = list(map(lambda tpl: tpl[2], full_melody_note_lengths))
        # round up - I don't want 0 length note lengths
        melody_note_lengths = list(
            map(
                lambda tpl: self.round_time(
                    tpl[1], True, self.__shortest_note < utils.SHORTEST_NOTE
                ),
                full_melody_note_lengths,
            )
        )

        melody_intervals = list()
        for idx in range(1, len(full_melody_note_lengths)):
            rounded_interval = (
                self.round_time(
                    full_melody_note_lengths[idx][0]
                    - full_melody_note_lengths[idx - 1][0],
                    False,
                    self.__shortest_note < utils.SHORTEST_NOTE,
                )
                - melody_note_lengths[idx - 1]
            )
            if rounded_interval < 0:
                rounded_interval = 0

            melody_intervals.append(rounded_interval)
        melody_intervals.append(0)

        melody_starts_of_bar = list(map(lambda tpl: tpl[3], full_melody_note_lengths))

        return melody_notes, melody_note_lengths, melody_intervals, melody_starts_of_bar

    def __count(self, dict: Dict[tuple, int], tuple: tuple) -> None:
        if dict.get(tuple) is not None:
            dict[tuple] += 1
        else:
            dict[tuple] = 1

    def __count_track_tuple_ngrams(
        self,
        tuples: List[Tuple[int]],
        ngrams: List[Tuple[int]],
        ngrams_without_octaves: List[Tuple[str, int, bool]],
        nminus1grams: List[Tuple[int]],
        nminus1grams_without_octaves: List[Tuple[str, int, bool]],
        starts_of_bar: List[bool] = None,
        nminus1gram_starts_of_bar: set[Tuple[int]] = None,
        nminus1gram_without_octaves_starts_of_bar: set[Tuple[int]] = None,
    ) -> None:
        for note_idx in range(len(tuples) - self.n + 2):
            # count n-1-gram
            nminus1gram = tuple(
                [tuples[note_idx + offset] for offset in range(self.n - 1)]
            )
            if note_idx > 0:
                prev_note = tuples[note_idx - 1][0]
            else:
                prev_note = -1
            nminus1gram_without_octaves = list()
            for idx in range(len(nminus1gram)):
                note, note_length, time = nminus1gram[idx]
                nminus1gram_without_octaves.append(
                    (
                        utils.get_note_name(note),
                        note_length,
                        time,
                        note > prev_note if note != prev_note else None,
                    )  # None if same note
                )
                prev_note = note
            nminus1gram_without_octaves = tuple(nminus1gram_without_octaves)

            if starts_of_bar is not None:
                if starts_of_bar[note_idx]:
                    nminus1gram_starts_of_bar.add(nminus1gram)
                    nminus1gram_without_octaves_starts_of_bar.add(
                        nminus1gram_without_octaves
                    )

            self.__count(nminus1grams, nminus1gram)
            self.__count(nminus1grams_without_octaves, nminus1gram_without_octaves)

            # not last n-1 tuples -> count n-gram
            if note_idx != len(tuples) - self.n + 1:
                ngram = nminus1gram + (tuples[note_idx + self.n - 1],)
                ngram_without_octaves = nminus1gram_without_octaves + (
                    (
                        utils.get_note_name(tuples[note_idx + self.n - 1][0]),
                        tuples[note_idx + self.n - 1][1],
                        tuples[note_idx + self.n - 1][2],
                        tuples[note_idx + self.n - 1][0]
                        > tuples[note_idx + self.n - 2][0]
                        if tuples[note_idx + self.n - 1][0]
                        != tuples[note_idx + self.n - 2][0]
                        else None,
                    ),
                )

                self.__count(ngrams, ngram)
                self.__count(ngrams_without_octaves, ngram_without_octaves)
