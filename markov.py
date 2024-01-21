import os
from mido import MidiFile, MetaMessage
from tqdm import tqdm
import math
from typing import List, Tuple, Dict, Set, Optional
import pickle

import utils


class MarkovModel:
    """Object representing a Markov model of given songs."""

    def __init__(
        self,
        n: int,
        dir: bool,
        pathname: str,
        merge_tracks: bool,
        ignore_bass: bool,
        max_tracks: int = None,
        key: Optional[str] = None,
        time_signature: Optional[str] = None,
        lengths_flatten_factor: Optional[int] = None,
        allow_major_minor_transpositions: bool = False,
    ) -> None:
        """
        Collects input MIDI sequence files and processes them - parses tracks with MIDI messages,
        counts n-grams and n-1-grams and extracts melody and chords for method 1.
        Notes saved in the model are first transposed to given key, if specified.
        """
        self.n = n  # length of n-grams

        # TUPLES (NOTE, NOTE LENGTH, TIME FROM NOTE START TO START OF NEXT NOTE) - MELODY WITH HARMONY
        self.harmony_ngrams = dict()
        self.harmony_nminus1grams = dict()
        # (NOTE, NOTE LENGTH, TIME FROM NOTE START TO START OF NEXT NOTE, IF NOTE PITCH GOES UP)
        self.harmony_ngrams_without_octaves = dict()
        self.harmony_nminus1grams_without_octaves = dict()

        # harmony n-1-grams which start some bar in input tracks
        self.harmony_nminus1gram_starts_of_bar = set()
        self.harmony_nminus1gram_without_octaves_starts_of_bar = set()

        # TUPLES (NOTE, NOTE LENGTH, TIME FROM NOTE END TO START OF NEXT NOTE) - MELODY
        self.melody_ngrams = dict()
        self.melody_nminus1grams = dict()
        # (NOTE, NOTE LENGTH, TIME FROM NOTE END TO START OF NEXT NOTE, IF NOTE PITCH GOES UP)
        self.melody_ngrams_without_octaves = dict()
        self.melody_nminus1grams_without_octaves = dict()

        # melody n-1-grams which start some bar in input tracks
        self.melody_nminus1gram_starts_of_bar = set()
        self.melody_nminus1gram_without_octaves_starts_of_bar = set()

        # dict: chord -> how many
        self.chords = dict()
        self.chords_without_octaves = dict()

        # TUPLES (NOTE, NOTE LENGTH, TIME/POSITION IN BAR)
        self.bar_ngrams = dict()
        self.bar_nminus1grams = dict()
        # (NOTE, NOTE LENGTH, TIME/POSITION IN BAR, IF NOTE PITCH GOES UP)
        self.bar_ngrams_without_octaves = dict()
        self.bar_nminus1grams_without_octaves = dict()

        # time signature
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

        # default: 60 ticks (32nd note length)
        self.length_precision = int(
            utils.DEFAULT_TICKS_PER_BEAT
            / (self.__shortest_note / utils.DEFAULT_BEAT_VALUE)
        )

        self.main_bar_length = int(
            self.main_beats_per_bar
            * (utils.DEFAULT_TICKS_PER_BEAT / (self.main_beat_value / 4))
        )
        # for bar n-grams
        self.__bar_lengths = list()
        self.__current_bar_length = None

        # whole note or bar length, if it's shorter than whole note (for convenience and faster generation)
        self.max_length = min(
            utils.DEFAULT_TICKS_PER_BEAT * 4,
            self.main_bar_length,
        )

        # determines if time signature is simple or compound,
        # and chooses set of used note lengths based on that
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

        # reading and processing MIDI sequences, writing tokens to nanoGPT text corpus
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

            mids = list()
            self.processed_mids = 0

            if dir and pathname == "serialized":
                print("Loading serialized MIDI...")
                mids = pickle.load(open(os.path.join(os.path.dirname(__file__), pathname), "rb"))
            else:
                self.__collect_mid_files(dir, mids)
                print("Serializing MIDI...")
                pickle.dump(mids, open(os.path.join(os.path.dirname(__file__), "serialized"), "wb"))

            print("Processing mid tracks and creating Markov model...")
            for mid in tqdm(mids):
                self.__process_mid_file(mid, merge_tracks, ignore_bass, max_tracks)

            # calculates the average tempo
            if self.main_tempo > 0:
                self.main_tempo //= self.__tempos_count
            else:
                self.main_tempo = utils.DEFAULT_TEMPO

            self.notes_list_file1.close()
            self.notes_list_file2.close()

    def __collect_mid_files(self, dir: bool, mids: List[MidiFile]) -> None:
        """Collects .mid file or files from dir."""
        print("Collecting and parsing mid files...")
        if dir:
            for root, _, files in os.walk(self.path):
                for filename in tqdm(files):
                    file = os.path.join(root, filename)
                    if (
                        os.path.isfile(file)
                        and os.path.splitext(file)[-1].lower() == ".mid"
                    ):
                        try:
                            mid_file = MidiFile(file)
                            if not mid_file.type == 2:
                                mids.append(mid_file)
                            else:
                                print(f"Skipped {mid_file.filename} - type 2!")
                        except Exception:
                            print(f"\nError - corrupted file: skipped {file}")
            if not mids:
                raise ValueError("No uncorrupted .mid files of type 0 or 1 in given directory!")
        else:  # assumes file is of .mid extension
            mid_file = MidiFile(self.path)
            if mid_file.type == 2:
                raise ValueError(".mid file should be of type 0 or 1!")
            mids.append(mid_file)

    def round_time(
        self,
        length: int,
        up: bool,
        in_time_signature: bool,
        lengths_flatten_factor: Optional[int] = None,
    ) -> int:
        """
        Rounds length up or down to length precision, which is multiplied by lengths_flatten_factor if specified.
        Then rounds up to closest note from used note lengths for given time signature, if in_time_signature is True.
        Clips the length to max length if rounded length is longer.
        """
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
        """
        Sorts the note lengths by start time, calculates distances between notes
        and then counts all the n-grams and n-1-grams. Appends notes to nanoGPT corpus.
        """
        if note_lengths:
            note_lengths = list(set(note_lengths))
            note_lengths.sort()
            notes = list(map(lambda tpl: tpl[2], note_lengths))

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
                melody_pauses,
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

            melody_tuples = list(zip(melody_notes, melody_note_lengths, melody_pauses))
            harmony_tuples = list(
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
            # for generate_with_harmony_ngrams
            self.__count_track_tuple_ngrams(
                harmony_tuples,
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

            self.notes_list_file1.write("START ")
            self.notes_list_file2.write("START ")
            # append to file for nanoGPT
            for note, note_length, until_next_note_start in harmony_tuples:
                self.notes_list_file1.write(
                    f"I{str(until_next_note_start)} N{str(note)} L{str(note_length)} "
                )
                self.notes_list_file2.write(
                    f"{str(note)},{str(note_length)},{str(until_next_note_start)} "
                )
            self.notes_list_file1.write("END\n")
            self.notes_list_file2.write("END\n")

    def __transpose_track(
        self, note_lengths: List[Tuple[int, bool]]
    ) -> List[Tuple[int]]:
        """Infers the key of the whole track and transposes it to key given in MarkovModel constructor."""
        notes_str = list(map(lambda tpl: utils.get_note_name(tpl[2]), note_lengths))
        self.__current_key = utils.infer_key(notes_str)
        if self.__current_key is None:
            return None

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
        self, mid: MidiFile, merge_tracks: bool, ignore_bass: bool, max_tracks: int
    ) -> int:
        """
        Iterates the tracks of the MIDI sequence, collects and counts notes,
        transposing them first to common key, if it was given in the MarkovModel constructor.
        Always ignores drums and percussive instruments, ignores bass tracks if ignore_bass is True.
        Merges tracks of different instruments if merge_tracks is True.
        """
        # to count lengths properly
        ticks_per_beat_factor = utils.DEFAULT_TICKS_PER_BEAT / mid.ticks_per_beat
        # list of pairs: (start in total ticks, key)
        self.__keys = list()
        # list of pairs: (start in total ticks, bar length)
        self.__bar_lengths = list()
        # dict: channel -> instrument
        instruments = {ch: 0 for ch in range(16)}

        self.__current_key = None
        self.__current_bar_length = (
            utils.DEFAULT_BEATS_PER_BAR * utils.DEFAULT_TICKS_PER_BEAT
        )
        self.__current_tempo = utils.DEFAULT_TEMPO

        if merge_tracks:
            all_note_lengths = list()
        processed_tracks = 0

        if self.main_key:
            first_notes_track = True

        for _, track in enumerate(mid.tracks):
            total_time = 0
            key_idx = 0
            meter_idx = 0

            # list of tuples (start of note, note length, note, if start of bar, time in bar)
            note_lengths = list()
            currently_playing_notes_starts = dict()

            for msg in track:
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

                                meter_start = 0
                                if self.__bar_lengths:
                                    meter_start = self.__bar_lengths[meter_idx][0]
                                time_in_bar = (
                                    int((start - meter_start) * ticks_per_beat_factor)
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

            if note_lengths:
                processed_tracks += 1
                if merge_tracks:
                    all_note_lengths += note_lengths
                else:
                    if (
                        self.main_key and not self.__current_key and first_notes_track
                    ):  # only happens on first track with notes
                        note_lengths = self.__transpose_track(note_lengths)
                        if note_lengths is None:
                            return  # skip file in training
                        first_notes_track = False
                    self.__count_all(note_lengths)

            if max_tracks is not None:
                if processed_tracks == max_tracks:
                    break

        if merge_tracks:
            if self.main_key and not self.__current_key and all_note_lengths:
                all_note_lengths = self.__transpose_track(all_note_lengths)
                if all_note_lengths is None:
                    return  # skip file in training
            self.__count_all(all_note_lengths)

        self.main_tempo += self.__current_tempo
        self.__tempos_count += 1
        self.processed_mids += 1

    def __read_meta_message(self, msg: MetaMessage, total_time: int) -> None:
        """Reads tempo, time signature or key from a MetaMessage."""
        if msg.type == "set_tempo":
            if self.__current_tempo:
                self.main_tempo += self.__current_tempo
                self.__tempos_count += 1
            self.__current_tempo = msg.tempo
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
        elif msg.type == "key_signature":
            self.__keys.append((total_time, msg.key))

    def __extract_melody_and_chords(
        self, note_lengths: List[Tuple[int, bool]]
    ) -> Tuple[List[int], bool]:
        """
        Extracts the track's melody, choosing the highest note in each time frame (box).
        Calculates pauses between notes in the melody.
        Extracts chords as tuples of notes playing simultaneously in a time frame (box).
        """

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

        melody_pauses = list()
        for idx in range(1, len(full_melody_note_lengths)):
            # round down - I want 0 length pauses
            rounded_pause = (
                self.round_time(
                    full_melody_note_lengths[idx][0]
                    - full_melody_note_lengths[idx - 1][0],
                    False,
                    self.__shortest_note < utils.SHORTEST_NOTE,
                )
                - melody_note_lengths[idx - 1]
            )
            if rounded_pause < 0:
                rounded_pause = 0

            melody_pauses.append(rounded_pause)
        melody_pauses.append(0)

        melody_starts_of_bar = list(map(lambda tpl: tpl[3], full_melody_note_lengths))

        return melody_notes, melody_note_lengths, melody_pauses, melody_starts_of_bar

    def __count(self, dict: Dict[Tuple, int], tuple: Tuple) -> None:
        """Increments the count of tuple in dict."""
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
        starts_of_bar: Optional[List[bool]] = None,
        nminus1gram_starts_of_bar: Optional[Set[Tuple[int]]] = None,
        nminus1gram_without_octaves_starts_of_bar: Optional[Set[Tuple[int]]] = None,
    ) -> None:
        """Counts the track's n-grams and n-1-grams of given type. Saves the n-1-grams which start a bar."""
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
