import os
import utils
from mido import MidiFile, MidiTrack, Message, MetaMessage, tempo2bpm
import numpy as np
import random

from markov import MarkovModel


class MusicGenerator:
    def __init__(self, mm: MarkovModel) -> None:
        self.mm = mm
        self.note_length_ppbs = np.array([])

    def __choose_next_note(self, prev_notes: tuple, with_octave=True) -> int | str:
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
            ngrams_count = ngrams.get(prev_notes + (note,))
            if ngrams_count is not None:
                ppbs.append(ngrams_count / nminus1grams[prev_notes])
            else:
                ppbs.append(0)
        if sum(ppbs) == 0:
            return None  # no such n-grams

        # normalization
        ppbs = np.array(ppbs)
        ppbs /= ppbs.sum()

        note_choice = np.random.choice(notes, p=ppbs)
        # if not with_octave:
        #     note_choice = utils.get_note_name(note_choice)
        return note_choice

    def __choose_next_length_from_ngrams(
        self, prev: tuple, if_note_length: bool
    ) -> int:
        if len(prev) != self.mm.m - 1:
            raise ValueError(
                f"With m-gram there has to be n-1 previous {'note lengths' if if_note_length else 'intervals'}!"
            )

        # 0 to 2 whole notes
        lengths_range = 2 * utils.DEFAULT_TICKS_PER_BEAT * self.mm.main_beat_value + 1
        valid_lengths = list(range(0, lengths_range, self.mm.length_precision))
        ppbs = []

        if if_note_length:
            ngrams = self.mm.note_length_ngrams
            nminus1grams = self.mm.note_length_nminus1grams
        else:
            ngrams = self.mm.interval_ngrams
            nminus1grams = self.mm.interval_nminus1grams

        for length in valid_lengths:
            ngrams_count = ngrams.get(prev + (length,))
            if ngrams_count is not None:
                ppbs.append(ngrams_count / nminus1grams[prev])
            else:
                ppbs.append(0)
        if sum(ppbs) == 0:
            return None  # no such m-grams

        # normalization
        ppbs = np.array(ppbs)
        ppbs /= ppbs.sum()

        length_choice = np.random.choice([i for i in valid_lengths], p=ppbs)
        return length_choice

    def __get_32nd_note_length(self, ticks_per_beat: int, beat_value: int) -> int:
        return ticks_per_beat // (32 // beat_value)

    def __calculate_note_length_ppbs(
        self, length_of_32nd: int, simple_time: bool, beat_value: int
    ) -> None:
        counts = self.mm.note_length_counts

        # MODIFIES LIST IN UTILS!
        if beat_value < 8:
            utils.note_lengths.pop(0)  # don't put 32nd notes in 2/2, 4/4, 3/4 etc.

        # round to used note lengths from utils
        rounded_counts = {k * length_of_32nd: 0 for k in utils.note_lengths}
        nearest_index = 0
        for note_length in sorted(list(counts.keys())):
            if note_length > length_of_32nd * utils.note_lengths[nearest_index]:
                nearest_index += 1
            if nearest_index == len(utils.note_lengths):
                break # don't count longer notes
            rounded_counts[
                utils.note_lengths[nearest_index] * length_of_32nd
            ] += counts[note_length]

        print(rounded_counts)
        # CONTROVERSIAL
        # as 8th, 16th and 32nd notes are generated in groups, let's not favour them that much
        if simple_time:
            # rounded_counts[length_of_32nd] //= 8
            rounded_counts[length_of_32nd * 2] //= 4
            rounded_counts[length_of_32nd * 4] //= 2
        else:
            # rounded_counts[length_of_32nd] //= 12
            rounded_counts[length_of_32nd * 2] //= 6
            rounded_counts[length_of_32nd * 4] //= 3

        all = sum(rounded_counts.values())
        ppbs = list(map(lambda x: x / all, list(rounded_counts.values())))

        # normalization
        self.note_length_ppbs = np.array(ppbs)
        self.note_length_ppbs /= self.note_length_ppbs.sum()

        print(self.note_length_ppbs)

    def __calculate_pause_ppb(self):
        # primitive way for now
        counts = self.mm.interval_counts
        all = sum(counts.values())
        return 1 - counts[0] / all

    def __choose_next_length_from_ppbs(self, length_of_32nd: int) -> int:
        length_choice = np.random.choice(
            [i * length_of_32nd for i in utils.note_lengths], p=self.note_length_ppbs
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
        # TODO: add custom key
        if self.mm.main_key != "":
            key = self.mm.main_key
            track.append(MetaMessage("key_signature", key=key))

    def __first_nminus1_notes(self, only_high_notes: bool) -> tuple[tuple, list]:
        # NGRAMS
        while True:
            prev_notes = random.choice(
                list(mm.note_nminus1grams.keys())
            )  # could be also parameterized
            first_notes = list(prev_notes)

            if only_high_notes:
                if all(map(lambda x: x >= utils.HIGH_NOTES_THRESHOLD, first_notes)):
                    break
            else:
                break
        return prev_notes, first_notes

    def __pick_note(
        self,
        first_notes: tuple[int | str],
        prev_notes: list[int | str],
        only_high_notes: bool,
    ) -> tuple[int | str, tuple[int | str]]:
        if first_notes:
            next_note = first_notes.pop(0)
            print(f"Chosen {next_note} note")
        else:
            while True:
                next_note = self.__choose_next_note(prev_notes, True)
                if next_note is None:
                    raise RuntimeError(
                        "Couldn't find next note and finish track. Try again or set smaller n."
                    )  # ugly error for now
                if only_high_notes:
                    if next_note >= utils.HIGH_NOTES_THRESHOLD:
                        break
                else:
                    break
            print(f"Chosen {next_note} note after {prev_notes}")
            prev_notes = prev_notes[1:] + (next_note,)
        return next_note, prev_notes

    def __print_track(self) -> None:
        print("Generated track:")
        test_mid = MidiFile(os.path.join(os.path.dirname(__file__), utils.OUTPUT_MID_FILE_NAME))
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
        first_notes: tuple[int | str],
        prev_notes: list[int | str],
        only_high_notes: bool,
        no_pauses: bool,
    ) -> list[tuple[int | str, int, bool]]:
        
        strong_beats = bar_length // strong_beat_length
        strong_beat, time_in_bar, time_in_strong_beat = 0, 0, 0
        notes = []  # consecutive notes as tuples: (note pitch, note length, if_pause)

        pause_ppb = self.__calculate_pause_ppb()
        print(f"Pause ppb: {pause_ppb}")
        note_or_pause_ppbs = [1 - pause_ppb, pause_ppb]
        if no_pauses:
            note_or_pause_ppbs = [1, 0]

        while strong_beat < strong_beats:
            note = -1
            note_length = self.__choose_next_length_from_ppbs(length_of_32nd)
            not_pause = np.random.choice([True, False], p=note_or_pause_ppbs)
            if time_in_strong_beat == 0:  # let's start the strong beat with a note
                not_pause = True

            # ends bar
            if time_in_bar + note_length == bar_length:
                if not_pause:
                    note, prev_notes = self.__pick_note(
                        first_notes, prev_notes, only_high_notes
                    )
                notes.append((note, note_length, not_pause))
                break

            if time_in_strong_beat + note_length > strong_beat_length:  # too long note
                continue
            else:
                if (
                    time_in_strong_beat + note_length == strong_beat_length
                ):  # ends strong beat part
                    if not_pause:
                        note, prev_notes = self.__pick_note(
                            first_notes, prev_notes, only_high_notes
                        )
                    notes.append((note, note_length, not_pause))
                    time_in_strong_beat = 0
                    strong_beat += 1
                    continue
                else:  # shorter than till end of strong beat part
                    if simple_time and note_length // length_of_32nd in [3, 6, 12, 24]:
                        continue  # don't put dotted notes in simple time (simplification, TODO: improve)
                    elif not simple_time and note_length // length_of_32nd in [8, 16]:
                        continue  # don't put half and quarter notes in compound time (simplification, TODO: improve)
                    elif note_length // length_of_32nd not in [
                        1,
                        2,
                        3,
                        4,
                        6,
                    ]:  # put one
                        if not_pause:
                            note, prev_notes = self.__pick_note(
                                first_notes, prev_notes, only_high_notes
                            )
                        notes.append((note, note_length, not_pause))
                        time_in_strong_beat += note_length
                        continue
                    else:  # 32nd, 16th or 8th note - make a group of them (8 32nd, 4 16h, 2 8th for simple time or 12 32nd, 6 16th, 3 8th for compound),
                        # less notes (but multiples of 2 for simple and 3 for compound) only if no space in strong beat part
                        # also in compound time: put 2 (if no space: 1) dotted 8th notes/4 (if no space: 2 or 1) dotted 16th notes
                        group_length = 8 if simple_time else 12

                        number_of_notes = group_length // (
                            note_length // length_of_32nd
                        )

                        while (
                            time_in_strong_beat + number_of_notes * note_length
                            > strong_beat_length
                        ):
                            number_of_notes //= 2
                        for i in range(number_of_notes):
                            note = -1
                            if not_pause:
                                note, prev_notes = self.__pick_note(
                                    first_notes, prev_notes, only_high_notes
                                )
                            notes.append((note, note_length, not_pause))
                            not_pause = np.random.choice(
                                [True, False], p=note_or_pause_ppbs
                            )

                        time_in_strong_beat += number_of_notes * note_length
                        if time_in_strong_beat == strong_beat_length:
                            strong_beat += 1
                            time_in_strong_beat = 0
        return notes

    def generate_music_in_time_signature(
        self,
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

        prev_notes, first_notes = self.__first_nminus1_notes(only_high_notes)

        interval = 0
        for bar in range(bars):
            # simplification: there are no notes spanning between bars
            bar_notes = self.__fit_bar(
                length_of_32nd,
                simple_time,
                bar_length,
                strong_beat_length,
                first_notes,
                prev_notes,
                only_high_notes,
                no_pauses,
            )
            for note, note_length, not_pause in bar_notes:
                if not_pause:
                    track.append(Message("note_on", note=note, time=int(interval)))
                    track.append(Message("note_off", note=note, time=int(note_length)))
                    interval = 0
                else:
                    interval += note_length

        new_mid.save(os.path.join(os.path.dirname(__file__), utils.OUTPUT_MID_FILE_NAME))

        self.__print_track()

    def generate_music_with_length_ngrams(
        self,
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

        prev_notes, first_notes = self.__first_nminus1_notes(only_high_notes)
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
        while bar < bars:
            next_note, prev_notes = self.__pick_note(
                first_notes, prev_notes, only_high_notes
            )
            track.append(Message("note_on", note=next_note, time=int(interval)))

            if first_note_lengths:
                note_length = first_note_lengths.pop(0)
                print(f"Chosen {note_length} note length")
            else:
                note_length = self.__choose_next_length_from_ngrams(
                    prev_note_lengths, True
                )
                if note_length is None:
                    raise RuntimeError(
                        "Couldn't find next note length and finish track. Try again or set smaller m."
                    )  # ugly error for now
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
                    raise RuntimeError(
                        "Couldn't find next interval and finish track. Try again or set smaller m."
                    )  # ugly error for now
                print(f"Chosen {interval} interval after {prev_intervals}")
                prev_intervals = prev_intervals[1:] + (interval,)

            bar += (time_in_bar + interval) // bar_length
            time_in_bar = (time_in_bar + interval) % bar_length

        new_mid.save(os.path.join(os.path.dirname(__file__), utils.OUTPUT_MID_FILE_NAME))

        self.__print_track()


# parse arguments - will be expanded and moved to main file
n = 4
m = 8
filename = "deb_clai.mid"

if n < 2:
    raise ValueError("n must be >= 2!")
if m < 2:
    raise ValueError("m must be >= 2!")

# if user doesn't set m, then make m = n
mm = MarkovModel(n, m, filename)

# MusicGenerator(mm).generate_music_with_length_ngrams(
#     bars=40, instrument=1, only_high_notes=False
# )

MusicGenerator(mm).generate_music_in_time_signature(
    bars=40, instrument=1, only_high_notes=False, no_pauses=False
)

# outdated - worked very rarely
# generate_music(mm, bars=10, instrument=1, use_time_signature=True, use_length_ngrams=True)
