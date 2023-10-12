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

    def __add_ppb(
        self,
        ppbs: list[float],
        ngrams: dict[tuple],
        nminus1grams: dict[tuple],
        prev: tuple,
        value: int | tuple[int],
    ) -> None:
        ngrams_count = ngrams.get(prev + (value,))
        if ngrams_count is not None:
            ppbs.append(ngrams_count / nminus1grams[prev])
        else:
            ppbs.append(0)

    def __normalize_ppbs(self, ppbs) -> list[float]:
        ppbs = np.array(ppbs)
        ppbs /= ppbs.sum()
        return ppbs

    def __choose_next_note(
        self, prev_notes: tuple[int | str], with_octave: bool
    ) -> int | str:
        if len(prev_notes) != self.mm.n - 1:
            raise ValueError("With n-gram there has to be n-1 previous notes!")

        ppbs = []
        if with_octave:
            notes = [i for i in range(128)]  # numbers
            ngrams = self.mm.note_ngrams
            nminus1grams = self.mm.note_nminus1grams
        else:
            notes = utils.notes  # note strings
            ngrams = self.mm.note_ngrams_without_octaves
            nminus1grams = self.mm.note_nminus1grams_without_octaves

        for note in notes:
            self.__add_ppb(ppbs, ngrams, nminus1grams, prev_notes, note)
        if sum(ppbs) == 0:
            return None  # no such n-grams

        ppbs = self.__normalize_ppbs(ppbs)

        note_choice = np.random.choice(notes, p=ppbs)
        return note_choice

    def __choose_next_tuple(
        self, prev_tuples: tuple[tuple[int]], with_octave: bool
    ) -> tuple[int]:
        if len(prev_tuples) != self.mm.n - 1:
            raise ValueError("With n-gram there has to be n-1 previous notes!")

        valid_lengths_range = range(
            0, self.mm.lengths_range + 1, self.mm.length_precision
        )
        ppbs = []
        # DOUBLE CHECK!
        if with_octave:
            tuples = [
                (i, j, k)
                for i in range(128)
                for j in valid_lengths_range
                for k in valid_lengths_range
            ]  # notes as numbers
            ngrams = self.mm.tuple_ngrams
            nminus1grams = self.mm.tuple_nminus1grams
        else:
            tuples = [
                (i, j, k)
                for i in utils.notes
                for j in valid_lengths_range
                for k in valid_lengths_range
            ]  # notes as strings
            ngrams = self.mm.tuple_ngrams_without_octaves
            nminus1grams = self.mm.tuple_nminus1grams_without_octaves

        for tuple in tuples:
            self.__add_ppb(ppbs, ngrams, nminus1grams, prev_tuples, tuple)
        if sum(ppbs) == 0:
            return None  # no such n-grams

        ppbs = self.__normalize_ppbs(ppbs)

        tuple_index_choice = np.random.choice(len(tuples), p=ppbs)
        return tuples[tuple_index_choice]

    def __choose_next_length_from_ngrams(
        self, prev_lengths: tuple[int], if_note_length: bool
    ) -> int:
        if len(prev_lengths) != self.mm.m - 1:
            raise ValueError(
                f"With m-gram there has to be n-1 previous {'note lengths' if if_note_length else 'intervals'}!"
            )

        # 0 to 2 whole notes
        valid_lengths = list(
            range(0, self.mm.lengths_range + 1, self.mm.length_precision)
        )
        ppbs = []

        if if_note_length:
            ngrams = self.mm.note_length_ngrams
            nminus1grams = self.mm.note_length_nminus1grams
        else:
            ngrams = self.mm.interval_ngrams
            nminus1grams = self.mm.interval_nminus1grams

        for length in valid_lengths:
            self.__add_ppb(ppbs, ngrams, nminus1grams, prev_lengths, length)
        if sum(ppbs) == 0:
            return None  # no such m-grams

        ppbs = self.__normalize_ppbs(ppbs)

        length_choice = np.random.choice([i for i in valid_lengths], p=ppbs)
        return length_choice

    def __get_32nd_note_length(self, ticks_per_beat: int, beat_value: int) -> int:
        return ticks_per_beat // (32 // beat_value)

    def __calculate_note_length_ppbs(
        self, length_of_32nd: int, simple_time: bool, beat_value: int
    ) -> None:
        counts = self.mm.note_length_counts

        if simple_time:
            self.note_lengths = utils.note_lengths_simple_time
        else:
            self.note_lengths = utils.note_lengths_compound_time

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

        print(rounded_counts)
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

        print(self.note_length_ppbs)

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

    def __set_tempo(self, track) -> int:
        # TODO: add custom tempo
        if self.mm.main_tempo != 0:
            tempo = self.mm.main_tempo
            track.append(MetaMessage("set_tempo", tempo=tempo))
        else:
            tempo = utils.DEFAULT_TEMPO
            track.append(MetaMessage("set_tempo"))  # default 500000
        return tempo

    def __set_key(self, track) -> str:
        # TODO: set key, add custom key
        # if self.mm.main_key != "":
        #     key = self.mm.main_key
        #     track.append(MetaMessage("key_signature", key=key))
        pass

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
        if only_high_notes:
            possible_notes = list(
                filter(
                    lambda note: utils.get_note_octave(note)
                    >= utils.HIGH_NOTES_OCTAVE_THRESHOLD,
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
        if with_octave:
            while True:
                prev_notes = random.choice(
                    list(mm.note_nminus1grams.keys())
                )  # could be also parameterized
                first_notes = list(prev_notes)
                if only_high_notes:
                    if all(
                        map(
                            lambda note: utils.get_note_octave(note)
                            >= utils.HIGH_NOTES_OCTAVE_THRESHOLD,
                            first_notes,
                        )
                    ):
                        break
                else:
                    break
        else:
            prev_notes = random.choice(
                list(mm.note_nminus1grams_without_octaves.keys())
            )  # could be also parameterized
            first_notes = list(prev_notes)
        return prev_notes, first_notes

    # very similar to first_nminus1_notes - generalize?
    def __first_nminus1_tuples(
        self, with_octave: bool, only_high_notes: bool
    ) -> tuple[tuple[tuple[int | str]], list[tuple[int | str]]]:
        if with_octave:
            while True:
                prev_tuples = random.choice(
                    list(mm.tuple_nminus1grams.keys())
                )  # could be also parameterized
                first_tuples = list(prev_tuples)
                if only_high_notes:
                    if all(
                        map(
                            lambda tuple: utils.get_note_octave(tuple[0])
                            >= utils.HIGH_NOTES_OCTAVE_THRESHOLD,
                            first_tuples,
                        )
                    ):
                        break
                else:
                    break
        else:
            prev_tuples = random.choice(
                list(mm.tuple_nminus1grams_without_octaves.keys())
            )  # could be also parameterized
            first_tuples = list(prev_tuples)
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
            while True:
                next_note = self.__choose_next_note(prev_notes, with_octave)
                if next_note is None:
                    # raise RuntimeError(
                    #     "Couldn't find next note and finish track. Try again or set smaller n."
                    # )  # ugly error for now
                    return None
                if with_octave and only_high_notes:
                    if (
                        utils.get_note_octave(next_note)
                        >= utils.HIGH_NOTES_OCTAVE_THRESHOLD
                    ):
                        break
                else:
                    break
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
        total_time: int,
    ) -> tuple[tuple[int], tuple[tuple[int]], int]:
        if first_tuples:
            next_tuple = first_tuples.pop(0)
            print(f"Chosen {next_tuple}")
        else:
            while True:
                next_tuple = self.__choose_next_tuple(prev_tuples, with_octave)
                if next_tuple is None:
                    # raise RuntimeError(
                    #     "Couldn't find next note and finish track. Try again or set smaller n."
                    # )  # ugly error for now
                    return None
                if with_octave and only_high_notes:
                    if (
                        utils.get_note_octave(next_tuple[0])
                        >= utils.HIGH_NOTES_OCTAVE_THRESHOLD
                    ):
                        break
                else:
                    break
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

        return next_tuple, prev_tuples, total_time + next_tuple[2]

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
                        first_notes, prev_notes, with_octave, only_high_notes, prev_note
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
        only_high_notes: bool = False,
        no_pauses: bool = False,
    ) -> None:
        new_mid = MidiFile(
            type=0, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT
        )  # one track
        track, tempo, key = self.__start_track(new_mid, instrument)

        # TIME SIGNATURE CHOICE - TODO: add custom time signature? close in function
        if (
            mm.main_beats_per_bar != 0 and mm.main_beat_value != 0
        ):  # input .mid had time signature specified
            beats_per_bar, beat_value = mm.main_beats_per_bar, mm.main_beat_value
            track.append(
                MetaMessage(
                    "time_signature",
                    numerator=mm.main_beats_per_bar,
                    denominator=mm.main_beat_value,
                )
            )
        else:
            beats_per_bar, beat_value = (
                utils.DEFAULT_BEATS_PER_BAR,
                utils.DEFAULT_BEAT_VALUE,
            )
            track.append(MetaMessage("time_signature"))  # default 4/4

        bar_length = beats_per_bar * new_mid.ticks_per_beat

        # time_in_bar, strong_beat, time_in_strong_beat = 0, 0, 0
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

        # for now use ngrams without octaves iff only_high_notes
        with_octave = False if only_high_notes else True
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
        only_high_notes: bool = False,
    ) -> None:
        new_mid = MidiFile(
            type=0, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT
        )  # one track
        track, tempo, key = self.__start_track(new_mid, instrument)

        # for generated music's length purpose
        bar_length = utils.DEFAULT_BEATS_PER_BAR * new_mid.ticks_per_beat
        time_in_bar = 0

        # for now use ngrams without octaves iff only_high_notes
        with_octave = False if only_high_notes else True
        prev_notes, first_notes = self.__first_nminus1_notes(
            with_octave, only_high_notes
        )

        prev_note_lengths = random.choice(
            list(mm.note_length_nminus1grams.keys())
        )  # could be also parameterized
        first_note_lengths = list(prev_note_lengths)
        prev_intervals = random.choice(
            list(mm.interval_nminus1grams.keys())
        )  # could be also parameterized
        first_intervals = list(prev_intervals)

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

            if first_note_lengths:
                note_length = first_note_lengths.pop(0)
                print(f"Chosen {note_length} note length")
            else:
                note_length = self.__choose_next_length_from_ngrams(
                    prev_note_lengths, True
                )
                if note_length is None:
                    # raise RuntimeError(
                    #     "Couldn't find next note length and finish track. Try again or set smaller m."
                    # )  # ugly error for now
                    print("Couldn't find next note length - ending track!")
                    break
                print(f"Chosen {note_length} note length after {prev_note_lengths}")
                prev_note_lengths = prev_note_lengths[1:] + (note_length,)

            bar += (time_in_bar + note_length) // bar_length
            time_in_bar = (time_in_bar + note_length) % bar_length

            track.append(Message("note_off", note=next_note, time=int(note_length)))

            if first_intervals:
                interval = first_intervals.pop(0)
                print(f"Chosen {interval} interval")
            else:
                interval = self.__choose_next_length_from_ngrams(prev_intervals, False)
                if interval is None:
                    # raise RuntimeError(
                    #     "Couldn't find next interval and finish track. Try again or set smaller m."
                    # )  # ugly error for now
                    print("Couldn't find next interval - ending track!")
                    break
                print(f"Chosen {interval} interval after {prev_intervals}")
                prev_intervals = prev_intervals[1:] + (interval,)

            bar += (time_in_bar + interval) // bar_length
            time_in_bar = (time_in_bar + interval) % bar_length

        new_mid.save(os.path.join(os.path.dirname(__file__), output_file))

        self.__print_track(output_file)

    def generate_music_with_tuple_ngrams(
        self,
        output_file: str,
        bars: int,
        instrument: int,
        only_high_notes: bool = False,
    ) -> None:
        new_mid = MidiFile(
            type=0, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT
        )  # one track
        track, tempo, key = self.__start_track(new_mid, instrument)

        # for generated music's length purpose
        bar_length = utils.DEFAULT_BEATS_PER_BAR * new_mid.ticks_per_beat

        # for now use ngrams without octaves iff only_high_notes
        with_octave = False if only_high_notes else True
        prev_tuples, first_tuples = self.__first_nminus1_tuples(
            with_octave, only_high_notes
        )

        messages = []  # list of tuples (absolute start time, note, if_note_on)
        # ADDING MESSAGES LOOP
        total_time = 0
        prev_note = -1
        while True:
            ret = self.__pick_tuple(
                first_tuples,
                prev_tuples,
                with_octave,
                only_high_notes,
                prev_note,
                total_time,
            )
            if ret is None:
                print("Couldn't find next note - ending track!")
                break
            next_tuple, prev_tuples, start_of_next = ret

            prev_note = next_tuple[0]
            messages.append((total_time, next_tuple[0], True))
            messages.append((total_time + next_tuple[1], next_tuple[0], False))

            if total_time // bar_length >= bars:
                break

            total_time = start_of_next

        # sort messages by start time, append them to track
        messages.sort()
        prev_abs_time = 0
        for message in messages:
            delta_time = message[0] - prev_abs_time
            if message[2]:
                track.append(Message("note_on", note=message[1], time=delta_time))
            else:
                track.append(Message("note_off", note=message[1], time=delta_time))
            prev_abs_time = message[0]

        new_mid.save(os.path.join(os.path.dirname(__file__), output_file))

        self.__print_track(output_file)


# parse arguments - will be expanded and moved to main file
n = 3
m = 4
# pathname = "ragtime.mid"

# or dirname - e.g. -d or --dir flag
pathname = "znane"

if n < 2:
    raise ValueError("n must be >= 2!")
if m < 2:
    raise ValueError("m must be >= 2!")

# if user doesn't set m, then make m = n
mm = MarkovModel(n, m, True, pathname)

generator = MusicGenerator(mm)

generator.generate_music_with_length_ngrams(
    output_file="test1.mid", bars=20, instrument=1, only_high_notes=False
)

generator.generate_music_in_time_signature(
    output_file="test2.mid",
    bars=20,
    instrument=1,
    only_high_notes=False,
    no_pauses=False,
)

generator.generate_music_with_tuple_ngrams(
    output_file="test3.mid",
    bars=40,
    instrument=1,
    only_high_notes=True,
)

# outdated - worked very rarely. Maybe will work nice if many input .mid files?
# generate_music(mm, bars=10, instrument=1, use_time_signature=True, use_length_ngrams=True)
