import os
import utils
from mido import MidiFile, MidiTrack, Message, MetaMessage, tempo2bpm
import numpy as np
import random

from markov import MarkovModel


class MusicGenerator:
    def __init__(self, mm: MarkovModel) -> None:
        self.mm = mm
        self.note_lengths = []
        self.note_length_ppbs = np.array([])
        self.pause_ppb = 0
        self.note_lengths_range = range(
            self.mm.length_precision, self.mm.max_length + 1, self.mm.length_precision
        )
        self.until_next_note_range = range(
            0, self.mm.max_length + 1, self.mm.length_precision
        )

    # ================================ CHOICE METHODS ===========================
    def __add_ppb(
        self,
        ppbs: list[float],
        ngrams: dict[tuple],
        nminus1grams_count: int,
        prev: tuple,
        value: int | tuple[int],
    ) -> None:
        ngrams_count = ngrams.get(prev + (value,))
        if ngrams_count:
            ppbs.append(ngrams_count / nminus1grams_count)
        else:
            ppbs.append(0)

    def __normalize_ppbs(self, ppbs) -> list[float]:
        ppbs = np.array(ppbs)
        ppbs /= ppbs.sum()
        return ppbs

    def __is_valid(
        self, note: int | str, with_octave: bool, only_high_notes: bool
    ) -> bool:
        if with_octave and only_high_notes:
            if utils.get_note_octave(note) < utils.HIGH_NOTES_OCTAVE_THRESHOLD:
                return False
        return True

    def __choose_next_note(
        self,
        prev_notes: tuple[int | str],
        with_octave: bool,
        only_high_notes: bool,
    ) -> int | str:
        if len(prev_notes) != self.mm.n - 1:
            raise ValueError("With n-gram there has to be n-1 previous notes!")

        ppbs = []
        if with_octave:
            notes = [i for i in range(128)]  # numbers
            ngrams = self.mm.note_ngrams
            nminus1grams = self.mm.note_nminus1grams
        else:
            notes = utils.NOTES  # note strings
            ngrams = self.mm.note_ngrams_without_octaves
            nminus1grams = self.mm.note_nminus1grams_without_octaves

        for note in notes:
            if self.__is_valid(note, with_octave, only_high_notes):
                self.__add_ppb(ppbs, ngrams, nminus1grams[prev_notes], prev_notes, note)
            else:
                ppbs.append(0)
        if sum(ppbs) == 0:
            # emergency: choose nminus1gram (without first) so that next iteration can choose nth
            nminus2grams_count = sum(
                filter(
                    lambda x: x is not None,
                    [nminus1grams.get(prev_notes[1:] + (note,)) for note in notes],
                )
            )
            ppbs = []
            for note in notes:
                if self.__is_valid(note, with_octave, only_high_notes):
                    self.__add_ppb(
                        ppbs, nminus1grams, nminus2grams_count, prev_notes, note
                    )
                else:
                    ppbs.append(0)
            if sum(ppbs) == 0:
                return None  # can't choose next note

        ppbs = self.__normalize_ppbs(ppbs)

        note_choice = np.random.choice(notes, p=ppbs)
        return note_choice

    def __choose_next_tuple(
        self,
        prev_tuples: tuple[tuple[int]],
        with_octave: bool,
        only_high_notes: bool,
        melody: bool,
    ) -> tuple[int]:
        if len(prev_tuples) != self.mm.n - 1:
            raise ValueError("With n-gram there has to be n-1 previous notes!")

        ppbs = []
        if with_octave:
            tuples = [
                (i, j, k)
                for i in range(128)
                for j in self.note_lengths_range
                for k in self.until_next_note_range
            ]  # notes as numbers
            if melody:
                ngrams = self.mm.melody_ngrams
                nminus1grams = self.mm.melody_nminus1grams
            else:
                ngrams = self.mm.tuple_ngrams
                nminus1grams = self.mm.tuple_nminus1grams
        else:
            tuples = [
                (i, j, k)
                for i in utils.NOTES
                for j in self.note_lengths_range
                for k in self.until_next_note_range
            ]  # notes as strings
            if melody:
                ngrams = self.mm.melody_ngrams_without_octaves
                nminus1grams = self.mm.melody_nminus1grams_without_octaves
            else:
                ngrams = self.mm.tuple_ngrams_without_octaves
                nminus1grams = self.mm.tuple_nminus1grams_without_octaves

        for tuple in tuples:
            if self.__is_valid(tuple[0], with_octave, only_high_notes):
                self.__add_ppb(
                    ppbs, ngrams, nminus1grams[prev_tuples], prev_tuples, tuple
                )
            else:
                ppbs.append(0)
        if sum(ppbs) == 0:
            # emergency: choose nminus1gram (without first) so that next iteration can choose nth
            nminus2grams_count = sum(
                filter(
                    lambda x: x is not None,
                    [nminus1grams.get(prev_tuples[1:] + (tuple,)) for tuple in tuples],
                )
            )
            ppbs = []
            for tuple in tuples:
                if self.__is_valid(tuple[0], with_octave, only_high_notes):
                    self.__add_ppb(
                        ppbs, nminus1grams, nminus2grams_count, prev_tuples, tuple
                    )
                else:
                    ppbs.append(0)
            if sum(ppbs) == 0:
                return None  # can't choose next tuple

        ppbs = self.__normalize_ppbs(ppbs)

        tuple_index_choice = np.random.choice(len(tuples), p=ppbs)
        tpl = tuples[tuple_index_choice]
        return tpl

    def __choose_next_lengths_from_ngrams(self, prev_lengths: tuple[tuple[int]]) -> int:
        if len(prev_lengths) != self.mm.n - 1:
            raise ValueError(f"With n-gram there has to be n-1 previous length_pairs!")

        ppbs = []
        length_pairs = [
            (i, j) for i in self.note_lengths_range for j in self.until_next_note_range
        ]
        for length_pair in length_pairs:
            self.__add_ppb(
                ppbs,
                self.mm.length_ngrams,
                self.mm.length_nminus1grams[prev_lengths],
                prev_lengths,
                length_pair,
            )
        if (
            sum(ppbs) == 0
        ):  # emergency: choose nminus1gram (without first) so that next iteration can choose nth
            nminus2grams_count = sum(
                filter(
                    lambda x: x is not None,
                    [
                        self.mm.length_nminus1grams.get(
                            prev_lengths[1:] + (length_pair,)
                        )
                        for length_pair in length_pairs
                    ],
                )
            )
            ppbs = []
            for length_pair in length_pairs:
                self.__add_ppb(
                    ppbs,
                    self.mm.length_nminus1grams,
                    nminus2grams_count,
                    prev_lengths,
                    length_pair,
                )
            if sum(ppbs) == 0:
                return None  # can't choose next length_pair

        ppbs = self.__normalize_ppbs(ppbs)

        lengths__index_choice = np.random.choice(len(length_pairs), p=ppbs)
        return length_pairs[lengths__index_choice]

    def __get_32nd_note_length(self, ticks_per_beat: int, beat_value: int) -> int:
        return ticks_per_beat // (32 // beat_value)

    def __calculate_note_length_ppbs(
        self, length_of_32nd: int, simple_time: bool, beat_value: int
    ) -> None:
        counts = self.mm.note_length_counts

        if simple_time:
            self.note_lengths = utils.NOTE_LENGTHS_SIMPLE_TIME
        else:
            self.note_lengths = utils.NOTE_LENGTHS_COMPOUND_TIME

        # MODIFIES LIST IN UTILS!
        if beat_value < 8:
            self.note_lengths.pop(0)  # don't put 32nd notes in 2/2, 4/4, 3/4 etc.

        # round to used note lengths from utils
        rounded_counts = {k * length_of_32nd: 0 for k in self.note_lengths}
        nearest_index = 0
        for note_length in sorted(list(counts.keys())):
            if note_length > length_of_32nd * self.note_lengths[nearest_index]:
                nearest_index += 1
            if nearest_index == len(self.note_lengths):
                break  # don't count longer notes
            rounded_counts[self.note_lengths[nearest_index] * length_of_32nd] += counts[
                note_length
            ]

        # print(rounded_counts)
        # CONTROVERSIAL
        # as 8th, 16th and 32nd notes are generated in groups, let's not favour them that much
        # TODO: improve
        # if simple_time:
        #     if beat_value >= 8:
        #         rounded_counts[length_of_32nd] //= 8
        #     rounded_counts[length_of_32nd * 2] //= 4
        #     rounded_counts[length_of_32nd * 4] //= 2
        # else:
        #     if beat_value >= 8:
        #         rounded_counts[length_of_32nd] //= 12
        #     rounded_counts[length_of_32nd * 2] //= 6
        #     rounded_counts[length_of_32nd * 4] //= 3

        all = sum(rounded_counts.values())
        ppbs = list(map(lambda cnts: cnts / all, list(rounded_counts.values())))

        self.note_length_ppbs = self.__normalize_ppbs(ppbs)

        # print(self.note_length_ppbs)

    def __calculate_pause_ppb(self) -> None:
        # primitive way for now
        counts = self.mm.interval_counts
        all = sum(counts.values())
        self.pause_ppb = 1 - (counts[0] + counts[self.mm.length_precision] / 2) / all

    def __choose_next_length_from_ppbs(self, length_of_32nd: int) -> int:
        length_choice = np.random.choice(
            [i * length_of_32nd for i in self.note_lengths], p=self.note_length_ppbs
        )
        return length_choice

    def __specific_first_note(self, note: str, only_high_notes: bool) -> int:
        low_end = (
            utils.HIGH_NOTES_OCTAVE_THRESHOLD
            if only_high_notes
            else utils.LOWEST_USED_OCTAVE
        )
        first_note = utils.get_note_in_octave(
            note, random.randint(low_end, utils.HIGHEST_USED_OCTAVE)
        )
        return first_note

    def __next_closest_note(
        self, prev_note: int, note: str, only_high_notes: bool
    ) -> int:
        # some randomness could be added - to not always take the closest note
        prev_note_octave = utils.get_note_octave(prev_note)
        possible_octaves = filter(
            lambda octave: octave <= utils.HIGHEST_USED_OCTAVE,
            [
                prev_note_octave,
                prev_note_octave - 1,
                prev_note_octave + 1,
            ],
        )

        possible_notes = [
            utils.get_note_in_octave(note, octave) for octave in possible_octaves
        ]
        threshold = (
            utils.HIGH_NOTES_OCTAVE_THRESHOLD
            if only_high_notes
            else utils.LOWEST_USED_OCTAVE
        )

        possible_notes = list(
            filter(
                lambda note: utils.get_note_octave(note) >= threshold,
                possible_notes,
            )
        )
        min_abs = 12
        for note in possible_notes:
            if abs(note - prev_note) < min_abs:
                offset = note - prev_note
                min_abs = abs(note - prev_note)
        return prev_note + offset

    def __first_nminus1_notes(
        self, with_octave: bool, only_high_notes: bool
    ) -> tuple[tuple[int | str], list[int | str]]:
        nminus1grams = (
            self.mm.note_nminus1grams
            if with_octave
            else self.mm.note_nminus1grams_without_octaves
        )
        while True:
            prev_notes = random.choice(
                list(nminus1grams.keys())
            )  # could be also parameterized
            first_notes = list(prev_notes)

            valid_notes = True
            for note in first_notes:
                valid_notes = valid_notes and self.__is_valid(
                    note, with_octave, only_high_notes
                )
            if valid_notes:
                break

        return prev_notes, first_notes

    def __first_nminus1_tuples(
        self, with_octave: bool, only_high_notes: bool, melody: bool
    ) -> tuple[tuple[tuple[int | str]], list[tuple[int | str]]]:
        if with_octave:
            nminus1grams = (
                self.mm.melody_nminus1grams if melody else self.mm.tuple_nminus1grams
            )
        else:
            nminus1grams = (
                self.mm.melody_nminus1grams_without_octaves
                if melody
                else self.mm.tuple_nminus1grams_without_octaves
            )
        while True:
            prev_tuples = random.choice(
                list(nminus1grams.keys())
            )  # could be also parameterized
            first_tuples = list(prev_tuples)

            valid_notes = True
            for tpl in first_tuples:
                valid_notes = valid_notes and self.__is_valid(
                    tpl[0], with_octave, only_high_notes
                )
            if valid_notes:
                break

        return prev_tuples, first_tuples

    def __pick_specific_note(
        self,
        first_notes: list[int | str],
        prev_notes: tuple[int | str],
        with_octave: bool,
        only_high_notes: bool,
        prev_note: int,
    ) -> tuple[int, tuple[int | str]]:
        if first_notes:
            next_note = first_notes.pop(0)
            print(f"Chosen {next_note} note")
        else:
            next_note = self.__choose_next_note(
                prev_notes, with_octave, only_high_notes
            )
            if next_note is None:
                # raise RuntimeError(
                #     "Couldn't find next note and finish track. Try again or set smaller n."
                # )  # ugly error for now
                return None
            print(f"Chosen {next_note} note after {prev_notes}")
            prev_notes = prev_notes[1:] + (next_note,)

        if not with_octave:  # calculate specific note
            if prev_note == -1:  # first note
                next_note = self.__specific_first_note(next_note, only_high_notes)
            else:
                next_note = self.__next_closest_note(
                    prev_note, next_note, only_high_notes
                )

        return next_note, prev_notes

    def __pick_tuple(
        self,
        first_tuples: list[tuple[int]],
        prev_tuples: tuple[tuple[int]],
        with_octave: bool,
        only_high_notes: bool,
        prev_note: int,
        melody: bool,
    ) -> tuple[tuple[int], tuple[tuple[int]]]:
        if first_tuples:
            next_tuple = first_tuples.pop(0)
            print(f"Chosen {next_tuple}")
        else:
            next_tuple = self.__choose_next_tuple(
                prev_tuples, with_octave, only_high_notes, melody
            )
            if next_tuple is None:
                # raise RuntimeError(
                #     "Couldn't find next note and finish track. Try again or set smaller n."
                # )  # ugly error for now
                return None
            print(f"Chosen {next_tuple} after {prev_tuples}")
            prev_tuples = prev_tuples[1:] + (next_tuple,)

        if not with_octave:  # calculate specific note
            if prev_note == -1:  # first note
                next_tuple = (
                    self.__specific_first_note(next_tuple[0], only_high_notes),
                    next_tuple[1],
                    next_tuple[2],
                )
            else:
                next_tuple = (
                    self.__next_closest_note(prev_note, next_tuple[0], only_high_notes),
                    next_tuple[1],
                    next_tuple[2],
                )

        return next_tuple, prev_tuples

    # =============================== TRACK METHODS ========================
    def __start_track(
        self, mid: MidiFile, instrument: int
    ) -> tuple[MidiTrack, int, int]:
        track = MidiTrack()
        mid.tracks.append(track)

        tempo = self.__set_tempo(track)
        key = self.__set_key(track)
        # INSTRUMENT
        track.append(Message("program_change", program=instrument, time=0))

        return track, tempo, key

    def __set_tempo(self, track: MidiTrack) -> int:
        # TODO: add custom tempo
        if self.mm.main_tempo != 0:
            tempo = self.mm.main_tempo
            track.append(MetaMessage("set_tempo", tempo=tempo))
        else:
            tempo = utils.DEFAULT_TEMPO
            track.append(MetaMessage("set_tempo"))  # default 500000
        return tempo

    def __set_key(self, track: MidiTrack) -> str | None:
        key = self.mm.main_key
        if key is not None:
            track.append(MetaMessage("key_signature", key=key))
        return key

    def __set_time_signature(self, track: MidiTrack) -> tuple[int]:
        # TODO: add custom time signature?
        if (
            self.mm.main_beats_per_bar != 0 and self.mm.main_beat_value != 0
        ):  # input .mid had time signature specified
            beats_per_bar, beat_value = (
                self.mm.main_beats_per_bar,
                self.mm.main_beat_value,
            )
            track.append(
                MetaMessage(
                    "time_signature",
                    numerator=self.mm.main_beats_per_bar,
                    denominator=self.mm.main_beat_value,
                )
            )
        else:
            beats_per_bar, beat_value = (
                utils.DEFAULT_BEATS_PER_BAR,
                utils.DEFAULT_BEAT_VALUE,
            )
            track.append(MetaMessage("time_signature"))  # default 4/4

        return beats_per_bar, beat_value

    def __print_track(self, output_file: str) -> None:
        print("Generated track:")
        test_mid = MidiFile(os.path.join(os.path.dirname(__file__), output_file))
        for track_idx, track in enumerate(test_mid.tracks):
            print(f"Track {track_idx}: {track.name}")
            for msg in track:
                print(msg)

    # ========================= MAIN GENERATING METHODS =================================================
    def __fit_bar(
        self,
        length_of_32nd: int,
        simple_time: bool,
        bar_length: int,
        strong_beat_length: int,
        first_notes: list[int | str],
        prev_notes: tuple[int | str],
        with_octave: bool,
        only_high_notes: bool,
        last_note: int,
    ) -> tuple[list[tuple[int | str, int, bool]], tuple[int | str], int]:
        strong_beats = bar_length // strong_beat_length
        strong_beat, time_in_bar, time_in_strong_beat = 0, 0, 0
        notes = []  # consecutive notes as tuples: (note pitch, note length, if_pause)
        # when pause: (-1, note_length, True)

        note_or_pause_ppbs = [1 - self.pause_ppb, self.pause_ppb]

        # long, could be rewritten equivalently shorter, but this way shows clearly all the options
        prev_note = last_note
        while strong_beat < strong_beats:
            next_note = -1
            note_length = self.__choose_next_length_from_ppbs(length_of_32nd)
            not_pause = np.random.choice([True, False], p=note_or_pause_ppbs)
            if time_in_strong_beat == 0:  # let's start the strong beat with a note
                not_pause = True

            # ends bar
            if time_in_bar + note_length == bar_length:
                if not_pause:
                    ret = self.__pick_specific_note(
                        first_notes,
                        prev_notes,
                        with_octave,
                        only_high_notes,
                        prev_note,
                    )
                    if ret is None:
                        return None
                    next_note, prev_notes = ret
                    prev_note = next_note
                notes.append((next_note, note_length, not_pause))
                break

            if time_in_strong_beat + note_length > strong_beat_length:  # too long note
                continue
            else:
                if (
                    time_in_strong_beat + note_length == strong_beat_length
                ):  # ends strong beat part
                    if not_pause:
                        ret = self.__pick_specific_note(
                            first_notes,
                            prev_notes,
                            with_octave,
                            only_high_notes,
                            prev_note,
                        )
                        if ret is None:
                            return None
                        next_note, prev_notes = ret

                        prev_note = next_note
                    notes.append((next_note, note_length, not_pause))
                    time_in_strong_beat = 0
                    strong_beat += 1
                    continue
                else:  # shorter than till end of strong beat part
                    # look in calculate_note_lengths_ppbs - forbids these lengths
                    # if simple_time and note_length // length_of_32nd in [3, 6, 12, 24]:
                    #     continue  # don't put dotted notes in simple time (simplification, TODO: improve)
                    # elif not simple_time and note_length // length_of_32nd in [8, 16]:
                    #     continue  # don't put half and quarter notes in compound time (simplification, TODO: improve)
                    if note_length // length_of_32nd not in [
                        1,
                        2,
                        3,
                        4,
                        6,
                    ]:  # put one
                        if not_pause:
                            ret = self.__pick_specific_note(
                                first_notes,
                                prev_notes,
                                with_octave,
                                only_high_notes,
                                prev_note,
                            )
                            if ret is None:
                                return None
                            next_note, prev_notes = ret

                            prev_note = next_note
                        notes.append((next_note, note_length, not_pause))
                        time_in_strong_beat += note_length
                        continue
                    else:  # 32nd, 16th or 8th note - make a group of them (8 32nd, 4 16h, 2 8th for simple time or 12 32nd, 6 16th, 3 8th for compound),
                        # less notes (but multiples of 2 for simple and 3 for compound) only if no space in strong beat part
                        # also in compound time: put 2 (if no space: 1) dotted 8th notes/4 (if no space: 2 or 1) dotted 16th notes
                        group_length = 8 if simple_time else 12

                        number_of_notes = group_length // (
                            note_length // length_of_32nd
                        )

                        while (  # smaller group if no space
                            time_in_strong_beat + number_of_notes * note_length
                            > strong_beat_length
                        ):
                            number_of_notes //= 2
                        for i in range(
                            number_of_notes
                        ):  # add group of notes and pauses
                            next_note = -1
                            if not_pause:
                                ret = self.__pick_specific_note(
                                    first_notes,
                                    prev_notes,
                                    with_octave,
                                    only_high_notes,
                                    prev_note,
                                )
                                if ret is None:
                                    return None
                                next_note, prev_notes = ret

                                prev_note = next_note
                            notes.append((next_note, note_length, not_pause))
                            not_pause = np.random.choice(
                                [True, False], p=note_or_pause_ppbs
                            )

                        time_in_strong_beat += number_of_notes * note_length
                        if time_in_strong_beat == strong_beat_length:
                            strong_beat += 1
                            time_in_strong_beat = 0
        return notes, prev_notes, prev_note

    def generate_music_in_time_signature(
        self,
        output_file: str,
        bars: int,
        instrument: int,
        with_octave: bool = True,
        only_high_notes: bool = False,
        no_pauses: bool = False,
    ) -> None:
        new_mid = MidiFile(
            type=0, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT
        )  # one track
        track, tempo, key = self.__start_track(new_mid, instrument)
        beats_per_bar, beat_value = self.__set_time_signature(track)

        bar_length = beats_per_bar * new_mid.ticks_per_beat

        strong_beat_length = bar_length
        simple_time = beats_per_bar in [2, 3, 4]
        # simple time signatures
        if beats_per_bar in [2, 4]:
            strong_beat_length //= 2  # beats_per_bar / 2
        # compound time signatures
        elif beats_per_bar in [6, 9, 12]:
            strong_beat_length //= beats_per_bar / 3
        # what about irregular time signatures?

        length_of_32nd = self.__get_32nd_note_length(new_mid.ticks_per_beat, beat_value)
        self.__calculate_note_length_ppbs(length_of_32nd, simple_time, beat_value)
        if not no_pauses:
            self.__calculate_pause_ppb()

        prev_notes, first_notes = self.__first_nminus1_notes(
            with_octave, only_high_notes
        )

        interval = 0
        last_note = -1
        for bar in range(bars):
            # simplification: there are no notes spanning between bars
            ret = self.__fit_bar(
                length_of_32nd,
                simple_time,
                bar_length,
                strong_beat_length,
                first_notes,
                prev_notes,
                with_octave,
                only_high_notes,
                last_note,
            )
            if ret is None:
                print("Couldn't find next note - ending track!")
                break

            bar_notes, prev_notes, last_note = ret

            for note, note_length, not_pause in bar_notes:
                if not_pause:
                    track.append(Message("note_on", note=note, time=int(interval)))
                    track.append(Message("note_off", note=note, time=int(note_length)))
                    interval = 0
                else:
                    interval += note_length

        new_mid.save(os.path.join(os.path.dirname(__file__), output_file))
        self.__print_track(output_file)

    def generate_music_with_length_ngrams(
        self,
        output_file: str,
        bars: int,
        instrument: int,
        with_octave: bool = True,
        only_high_notes: bool = False,
    ) -> None:
        new_mid = MidiFile(
            type=0, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT
        )  # one track
        track, tempo, key = self.__start_track(new_mid, instrument)

        # for generated music's length purpose
        bar_length = utils.DEFAULT_BEATS_PER_BAR * new_mid.ticks_per_beat
        time_in_bar = 0

        prev_notes, first_notes = self.__first_nminus1_notes(
            with_octave, only_high_notes
        )

        prev_lengths = random.choice(
            list(self.mm.length_nminus1grams.keys())
        )  # could be also parameterized
        first_lengths = list(prev_lengths)

        # MUSIC GENERATION LOOP
        interval = 0
        bar = 0
        prev_note = -1
        while bar < bars:
            ret = self.__pick_specific_note(
                first_notes, prev_notes, with_octave, only_high_notes, prev_note
            )
            if ret is None:
                print("Couldn't find next note - ending track!")
                break
            next_note, prev_notes = ret

            track.append(Message("note_on", note=next_note, time=int(interval)))
            prev_note = next_note

            if first_lengths:
                length_pair = first_lengths.pop(0)
                note_length, interval = length_pair
                print(f"Chosen {length_pair} lengths")
            else:
                length_pair = self.__choose_next_lengths_from_ngrams(prev_lengths)
                if length_pair is None:
                    # raise RuntimeError(
                    #     "Couldn't find next note length and interval and finish track. Try again or set smaller m."
                    # )  # ugly error for now
                    print("Couldn't find next note length and interval - ending track!")
                    break
                note_length, interval = length_pair
                print(f"Chosen {length_pair} lengths after {prev_lengths}")
                prev_lengths = prev_lengths[1:] + (length_pair,)

            track.append(Message("note_off", note=next_note, time=int(note_length)))

            bar += (time_in_bar + note_length + interval) // bar_length
            time_in_bar = (time_in_bar + note_length + interval) % bar_length

        new_mid.save(os.path.join(os.path.dirname(__file__), output_file))
        self.__print_track(output_file)

    def generate_music_with_melody_ngrams(
        self,
        output_file: str,
        bars: int,
        instrument: int,
        with_octave: bool = True,
        only_high_notes: bool = False,
    ) -> None:
        new_mid = MidiFile(
            type=0, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT
        )  # one track
        track, tempo, key = self.__start_track(new_mid, instrument)

        # for generated music's length purpose
        bar_length = utils.DEFAULT_BEATS_PER_BAR * new_mid.ticks_per_beat
        time_in_bar = 0

        prev_tuples, first_tuples = self.__first_nminus1_tuples(
            with_octave, only_high_notes, True
        )

        # MUSIC GENERATION LOOP
        interval = 0
        bar = 0
        prev_note = -1
        while bar < bars:
            ret = self.__pick_tuple(
                first_tuples,
                prev_tuples,
                with_octave,
                only_high_notes,
                prev_note,
                True,
            )
            if ret is None:
                print("Couldn't find next note - ending track!")
                break
            next_tuple, prev_tuples = ret
            next_note = next_tuple[0]

            track.append(Message("note_on", note=next_note, time=int(interval)))
            prev_note = next_note

            _, note_length, interval = next_tuple
            track.append(Message("note_off", note=next_note, time=int(note_length)))

            bar += (time_in_bar + note_length + interval) // bar_length
            time_in_bar = (time_in_bar + note_length + interval) % bar_length

        new_mid.save(os.path.join(os.path.dirname(__file__), output_file))
        self.__print_track(output_file)

    def __append_messages(self, track: MidiTrack, messages: list[tuple]) -> None:
        # sort messages by start time, append them to track
        messages.sort()
        prev_abs_time = 0
        for message in messages:
            start, note, is_note_on = message
            delta_time = start - prev_abs_time
            if is_note_on:
                track.append(Message("note_on", note=note, time=delta_time))
            else:
                track.append(Message("note_off", note=note, time=delta_time))
            prev_abs_time = message[0]

    def generate_music_with_tuple_ngrams(
        self,
        output_file: str,
        bars: int,
        instrument: int,
        with_octave: bool = True,
        only_high_notes: bool = False,
    ) -> None:
        new_mid = MidiFile(
            type=0, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT
        )  # one track
        track, tempo, key = self.__start_track(new_mid, instrument)

        # for generated music's length purpose
        bar_length = utils.DEFAULT_BEATS_PER_BAR * new_mid.ticks_per_beat

        prev_tuples, first_tuples = self.__first_nminus1_tuples(
            with_octave, only_high_notes, False
        )

        messages = []  # list of tuples (absolute start time, note, if_note_on)
        # ADDING MESSAGES LOOP
        total_time = 0
        prev_note = -1
        chord = set()
        while True:
            ret = self.__pick_tuple(
                first_tuples,
                prev_tuples,
                with_octave,
                only_high_notes,
                prev_note,
                False,
            )
            if ret is None:
                print("Couldn't find next note - ending track!")
                break
            next_tuple, prev_tuples = ret
            next_note, note_length, until_next_note_start = next_tuple

            if until_next_note_start == 0 and len(chord) == utils.MAX_CHORD_SIZE - 1:
                # start new chord
                until_next_note_start = utils.UNTIL_NEXT_CHORD * note_length

            # ugly, changes octave if the note is doubled
            if not with_octave:
                jump = 12
                trials = 1
                # look in "neighbouring" octaves
                while next_note in chord or next_note not in utils.NOTES_RANGE:
                    next_note += jump
                    jump = -(jump + 12) if jump > 0 else -(jump - 12)
                    trials += 1
                    if trials == 2 * utils.OCTAVES:
                        print("Couldn't find next note - ending track!")
                        return None

            if next_note not in chord:
                messages.append((total_time, next_note, True))
                messages.append((total_time + note_length, next_note, False))
            prev_note = next_note

            if until_next_note_start == 0:
                chord.add(next_note)
            else:
                chord = {next_note}

            if total_time // bar_length >= bars:
                break

            total_time += until_next_note_start

        self.__append_messages(track, messages)

        new_mid.save(os.path.join(os.path.dirname(__file__), output_file))
        self.__print_track(output_file)

    def generate_music_from_file_nanogpt(
        self,
        input_filepath: str,
        output_file: str,
        instrument: int,
    ) -> None:
        new_mid = MidiFile(
            type=0, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT
        )  # one track
        track, tempo, key = self.__start_track(new_mid, instrument)

        # for generated music's length purpose
        bar_length = utils.DEFAULT_BEATS_PER_BAR * new_mid.ticks_per_beat

        with open(input_filepath) as f:
            tuples = f.read().split()

        messages = []  # list of tuples (absolute start time, note, if_note_on)
        # ADDING MESSAGES LOOP
        chord = set()
        total_time = 0
        for tuple in tuples:
            next_note, note_length, until_next_note_start = map(int, map(lambda x: x[1:], tuple.split(",")))
            if until_next_note_start == 0 and len(chord) == utils.MAX_CHORD_SIZE - 1:
                # start new chord
                until_next_note_start = utils.UNTIL_NEXT_CHORD * note_length

            if next_note not in chord:
                messages.append((total_time, next_note, True))
                messages.append((total_time + note_length, next_note, False))

            if until_next_note_start == 0:
                chord.add(next_note)
            else:
                chord = {next_note}

            total_time += until_next_note_start

        self.__append_messages(track, messages)

        new_mid.save(os.path.join(os.path.dirname(__file__), output_file))
        self.__print_track(output_file)

# parse arguments - will be expanded and moved to main file
n = 3
if n < 2:
    raise ValueError("n must be >= 2!")

# single file
# pathname = "usa.mid"
# mm = MarkovModel(
#     n=n, dir=False, pathname=pathname, merge_tracks=True, ignore_bass=True, key="D"
# )

# or dirname - e.g. -d or --dir flag
pathname = "chopin_big"
mm = MarkovModel(
    n=n, dir=True, pathname=pathname, merge_tracks=True, ignore_bass=True, key="C"
)

if mm.processed_mids == 0:
    raise ValueError("Couldn't process any mids! Try turning off key signature.")

generator = MusicGenerator(mm)

if __name__ == '__main__':
    generator.generate_music_in_time_signature(
        output_file="test1.mid",
        bars=20,
        instrument=0,
        with_octave=True,
        only_high_notes=False,
        no_pauses=False,
    )

    generator.generate_music_with_length_ngrams(
        output_file="test2.mid",
        bars=20,
        instrument=0,
        with_octave=True,
        only_high_notes=False,
    )

    generator.generate_music_with_melody_ngrams(
        output_file="test3.mid",
        bars=20,
        instrument=0,
        with_octave=True,
        only_high_notes=False,
    )

    generator.generate_music_with_tuple_ngrams(
        output_file="test4.mid",
        bars=40,
        instrument=0,
        with_octave=True,
        only_high_notes=False,
    )

    generator.generate_music_from_file_nanogpt(
        input_filepath="nanoGPT/test.txt",
        output_file="test_gpt.mid",
        instrument=0
    )
