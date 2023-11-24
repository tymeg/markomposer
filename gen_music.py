import os
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
import numpy as np
import random
import math
from itertools import permutations
from typing import Dict

import utils
from markov import MarkovModel


class MusicGenerator:
    def __init__(
        self,
        mm: MarkovModel,
        k: int = None,
        p: float = None,
        uniform: bool = False,
        weighted_random_start: bool = False,
    ) -> None:
        self.mm = mm
        self.k = k  # for top-k sampling (greedy for k=1) - sample from k most probable
        self.p = p  # for top-p sampling - sample from the smallest possible set
        # whose cumulative probability exceeds the probability p
        self.uniform = uniform  # for uniform distribution sampling
        # (at most one is not None/True)

        self.weighted_random_start = weighted_random_start

        self.note_lengths_range = range(
            self.mm.length_precision, self.mm.max_length + 1, self.mm.length_precision
        )
        self.until_next_note_range = range(
            0, self.mm.max_length + 1, self.mm.length_precision
        )

    # ================================ CHOICE METHODS ===========================
    def __add_ppb(
        self,
        ppbs: Dict[tuple, float],
        ngrams: Dict[tuple, int],
        nminus1gram_count: int,
        prev: tuple,
        value: tuple,
    ) -> None:
        ngram = prev + (value,)
        ngram_count = ngrams.get(ngram)
        if ngram_count:
            if self.uniform:
                ppbs[value] = 1.0
            else:
                ppbs[value] = ngram_count / nminus1gram_count
        else:
            ppbs[value] = 0.0

    def __normalize_ppbs(self, ppbs: Dict[tuple, float]) -> np.ndarray[float]:
        # make list only of ppbs
        ppbs = list(ppbs.values())

        # normalize
        ppbs = np.array(ppbs, dtype="float64")
        ppbs /= ppbs.sum()
        return ppbs

    def __sampling(self, ppbs: Dict[tuple, float]):
        # sort dict by value
        ppbs_from_highest = {
            k: v
            for k, v in sorted(ppbs.items(), key=lambda item: item[1], reverse=True)
        }
        if self.k:  # top-k sampling
            if self.k >= len(ppbs):
                return ppbs
            for key in list(ppbs_from_highest.keys())[self.k :]:
                ppbs[key] = 0.0
        elif self.p:  # top-p sampling
            cumulative_ppb = 0.0
            count = 0
            for key in ppbs_from_highest.keys():
                cumulative_ppb += ppbs_from_highest[key]
                count += 1
                if cumulative_ppb >= self.p:
                    break
            for key in list(ppbs_from_highest.keys())[count:]:
                ppbs[key] = 0.0

    def __is_valid(
        self, note: int | str, with_octave: bool, only_high_notes: bool
    ) -> bool:
        if with_octave and only_high_notes:
            if utils.get_note_octave(note) < utils.HIGH_NOTES_OCTAVE_THRESHOLD:
                return False
        return True

    def __choose_next_tuple(
        self,
        prev_tuples: tuple[tuple[int] | tuple[int, bool]],
        with_octave: bool,
        only_high_notes: bool,
        type: int,
    ) -> tuple[int] | None:
        if len(prev_tuples) != self.mm.n - 1:
            raise ValueError("With n-gram there has to be n-1 previous notes!")

        # dict: tuple -> ppb of tuple
        ppbs = dict()
        if with_octave:
            tuples = [
                (i, j, k)
                for i in range(128)
                for j in self.note_lengths_range
                for k in self.until_next_note_range
            ]  # notes as numbers
            if type == 0:
                ngrams = self.mm.melody_ngrams
                nminus1grams = self.mm.melody_nminus1grams
            elif type == 1:
                ngrams = self.mm.tuple_ngrams
                nminus1grams = self.mm.tuple_nminus1grams
            elif type == 2:
                ngrams = self.mm.bar_ngrams
                nminus1grams = self.mm.bar_nminus1grams
        else:
            tuples = [
                (i, j, k, l)
                for i in utils.NOTES
                for j in self.note_lengths_range
                for k in self.until_next_note_range
                for l in [True, False]
            ]  # notes as strings
            if type == 0:
                ngrams = self.mm.melody_ngrams_without_octaves
                nminus1grams = self.mm.melody_nminus1grams_without_octaves
            elif type == 1:
                ngrams = self.mm.tuple_ngrams_without_octaves
                nminus1grams = self.mm.tuple_nminus1grams_without_octaves
            elif type == 2:
                ngrams = self.mm.bar_ngrams_without_octaves
                nminus1grams = self.mm.bar_nminus1grams_without_octaves

        for tuple in tuples:
            if self.__is_valid(tuple[0], with_octave, only_high_notes):
                self.__add_ppb(
                    ppbs, ngrams, nminus1grams[prev_tuples], prev_tuples, tuple
                )
            else:
                ppbs[tuple] = 0.0
        if sum(ppbs.values()) == 0.0:
            return None  # can't choose next tuple

        if self.k or self.p:
            self.__sampling(ppbs)

        ppbs = self.__normalize_ppbs(ppbs)

        tuple_index_choice = np.random.choice(len(tuples), p=ppbs)
        tpl = tuples[tuple_index_choice]
        return tpl

    def __specific_first_note(
        self, note: str, only_high_notes: bool, only_low_notes: bool = False
    ) -> int:
        low_end = (
            utils.HIGH_NOTES_OCTAVE_THRESHOLD
            if only_high_notes
            else utils.LOWEST_USED_OCTAVE
        )
        high_end = (
            utils.HIGH_NOTES_OCTAVE_THRESHOLD
            if only_low_notes
            else utils.HIGHEST_USED_OCTAVE
        )
        first_note = utils.get_note_in_octave(note, random.randint(low_end, high_end))
        return first_note

    def __next_closest_note(
        self,
        prev_note: int,
        note: str,
        go_up: bool,
        only_high_notes: bool,
        only_low_notes: bool = False,
    ) -> int:
        prev_note_octave = utils.get_note_octave(prev_note)
        possible_octaves = filter(
            lambda octave: utils.LOWEST_USED_OCTAVE
            <= octave
            <= utils.HIGHEST_USED_OCTAVE,
            [
                prev_note_octave,
                prev_note_octave - 1,
                prev_note_octave + 1,
            ],
        )

        possible_notes = [
            utils.get_note_in_octave(note, octave) for octave in possible_octaves
        ]
        if only_high_notes or only_low_notes:
            threshold = utils.HIGH_NOTES_OCTAVE_THRESHOLD
            possible_notes = list(
                filter(
                    lambda note: (
                        not only_high_notes or utils.get_note_octave(note) >= threshold
                    )
                    and (
                        not only_low_notes or utils.get_note_octave(note) <= threshold
                    ),
                    possible_notes,
                )
            )
        if go_up is None:  # same note
            return prev_note
        elif go_up:
            higher_notes = list(filter(lambda note: note > prev_note, possible_notes))
            if higher_notes:
                return min(higher_notes)
        else:
            lower_notes = list(filter(lambda note: note < prev_note, possible_notes))
            if lower_notes:
                return max(lower_notes)

        # not possible to pick the note Markov's chain advised to pick
        min_abs = 13
        for note in possible_notes:
            if abs(note - prev_note) < min_abs:
                offset = note - prev_note
                min_abs = abs(note - prev_note)
        return prev_note + offset

    def __first_nminus1_tuples(
        self,
        with_octave: bool,
        only_high_notes: bool,
        type: int,
        start_of_bar: bool = None,
        start_with_chord: bool = False,
        first_note: int | str = None,
    ) -> tuple[tuple[tuple[int | str]], list[tuple[int | str]]] | None:
        if with_octave:
            if type == 0:
                nminus1grams = self.mm.melody_nminus1grams
                starts_of_bar = self.mm.melody_nminus1gram_starts_of_bar
            elif type == 1:
                nminus1grams = self.mm.tuple_nminus1grams
                starts_of_bar = self.mm.tuple_nminus1gram_starts_of_bar
            elif type == 2:
                nminus1grams = self.mm.bar_nminus1grams
        else:
            if type == 0:
                nminus1grams = self.mm.melody_nminus1grams_without_octaves
                starts_of_bar = self.mm.melody_nminus1gram_without_octaves_starts_of_bar
            elif type == 1:
                nminus1grams = self.mm.tuple_nminus1grams_without_octaves
                starts_of_bar = self.mm.tuple_nminus1gram_without_octaves_starts_of_bar
            elif type == 2:
                nminus1grams = self.mm.bar_nminus1grams_without_octaves
        nminus1grams_keys = list(nminus1grams.keys())
        # optional?
        if start_of_bar:
            if type == 1:
                nminus1grams_keys = list(
                    filter(
                        lambda nminus1gram: nminus1gram in starts_of_bar,
                        nminus1grams_keys,
                    )
                )
            elif type == 2:
                nminus1grams_keys = list(
                    filter(
                        lambda nminus1gram: nminus1gram[0][2] == 0, nminus1grams_keys
                    )
                )

        if start_with_chord:
            if type == 1:
                nminus1grams_keys = list(
                    filter(
                        lambda nminus1gram: nminus1gram[0][2] == 0, nminus1grams_keys
                    )
                )
            elif type == 2 and len(nminus1grams_keys[0]) >= 2:
                nminus1grams_keys = list(
                    filter(
                        lambda nminus1gram: nminus1gram[0][2] == nminus1gram[1][2],
                        nminus1grams_keys,
                    )
                )

        if first_note is not None:
            # for continuation when can't pick nth note
            if isinstance(first_note, int) and with_octave:
                nminus1grams_keys = list(
                    filter(  # specific note
                        lambda tuples: tuples[0][0] == first_note,
                        nminus1grams_keys,
                    )
                )
            else:
                nminus1grams_keys = list(
                    filter(  # note name
                        lambda tuples: (
                            tuples[0][0]
                            if not with_octave
                            else utils.get_note_name(tuples[0][0])
                        )
                        == first_note,
                        nminus1grams_keys,
                    )
                )
            if not nminus1grams_keys or not any(
                [  # no nminus1gram consists only of valid notes
                    all(
                        [
                            self.__is_valid(tpl[0], with_octave, only_high_notes)
                            for tpl in nminus1grams_keys[i]
                        ]
                    )
                    for i in range(len(nminus1grams_keys))
                ]
            ):
                return None
        if self.weighted_random_start:
            ppbs = {
                nminus1gram: nminus1grams[nminus1gram]
                for nminus1gram in nminus1grams_keys
            }
            ppbs = self.__normalize_ppbs(ppbs)

        while True:
            if self.weighted_random_start:
                tuple_index_choice = np.random.choice(len(nminus1grams_keys), p=ppbs)
                prev_tuples = nminus1grams_keys[tuple_index_choice]
            else:
                prev_tuples = random.choice(nminus1grams_keys)
            first_tuples = list(prev_tuples)

            if all(
                [
                    self.__is_valid(tpl[0], with_octave, only_high_notes)
                    for tpl in first_tuples
                ]
            ):
                break

        return prev_tuples, first_tuples

    def __pick_tuple(
        self,
        first_tuples: list[tuple[int]],
        prev_tuples: tuple[tuple[int]],
        with_octave: bool,
        only_high_notes: bool,
        prev_note: int,
        type: int,
        time_in_bar: int,
        bar_length: int,
        messages: list[tuple[int, bool]] = None,
    ) -> tuple[tuple[int], tuple[tuple[int]]]:
        if first_tuples:
            next_tuple = first_tuples.pop(0)
            print(f"Chosen {next_tuple}")
        else:
            next_tuple = self.__choose_next_tuple(
                prev_tuples, with_octave, only_high_notes, type
            )
            if next_tuple is None:
                # "new start"
                # try finding nminus1gram with the first tuple as last generated
                if messages:
                    ret = self.__first_nminus1_tuples(
                        with_octave,
                        only_high_notes,
                        type,
                        start_of_bar=messages[-1][0] % bar_length == 0,
                        first_note=messages[-1][1],
                    )
                else:
                    ret = self.__first_nminus1_tuples(
                        with_octave,
                        only_high_notes,
                        type,
                        start_of_bar=(
                            time_in_bar - prev_tuples[-1][2] - prev_tuples[-1][1]
                        )
                        % bar_length
                        == 0,
                        first_note=prev_tuples[-1][0],
                    )
                if ret is not None and len(ret[0]) > 1:
                    prev_tuples, _ = ret
                    first_tuples.extend(prev_tuples[1:])
                else:
                    prev_tuples, _ = self.__first_nminus1_tuples(
                        with_octave,
                        only_high_notes,
                        type,
                        start_of_bar=time_in_bar == 0,
                    )
                    first_tuples.extend(prev_tuples)
                next_tuple = first_tuples.pop(0)
                print(f"Chosen {next_tuple}")
            else:
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
                    self.__next_closest_note(
                        prev_note, next_tuple[0], next_tuple[3], only_high_notes
                    ),
                    next_tuple[1],
                    next_tuple[2],
                )

        return next_tuple, prev_tuples

    # =============================== TRACK METHODS ========================
    def __start_track(
        self,
        mid: MidiFile,
        instrument: int,
        melody: bool,
    ) -> MidiTrack:
        track = MidiTrack()
        mid.tracks.append(track)

        # INSTRUMENT
        if melody:
            track.append(
                Message("program_change", channel=0, program=instrument, time=0)
            )
        else:
            track.append(
                Message("program_change", channel=1, program=instrument, time=0)
            )

        return track

    def __set_tempo(self, track: MidiTrack, tempo: int | None) -> None:
        if tempo is not None:
            tempo = bpm2tempo(tempo)
        else:
            tempo = self.mm.main_tempo
        track.append(MetaMessage("set_tempo", tempo=tempo))

    def __set_key(self, track: MidiTrack) -> None:
        key = self.mm.main_key
        if key is not None:
            track.append(MetaMessage("key_signature", key=key))

    def __set_time_signature(self, track: MidiTrack) -> tuple[int]:
        beats_per_bar, beat_value = self.mm.main_beats_per_bar, self.mm.main_beat_value
        track.append(
            MetaMessage(
                "time_signature",
                numerator=beats_per_bar,
                denominator=beat_value,
            )
        )

        return beats_per_bar, beat_value

    def __print_track(self, output_file: str) -> None:
        print("Generated track:")
        test_mid = MidiFile(os.path.join(os.path.dirname(__file__), output_file))
        for track_idx, track in enumerate(test_mid.tracks):
            print(f"Track {track_idx}: {track.name}")
            for msg in track:
                print(msg)

    # ========================= MAIN GENERATING METHODS =================================================
    def __calculate_strong_beat_length(
        self, bar_length: int, beats_per_bar: int
    ) -> int:
        strong_beat_length = bar_length
        # simple time signatures
        if beats_per_bar in [2, 4]:
            strong_beat_length //= beats_per_bar // 2
        # compound time signatures
        elif beats_per_bar in [6, 9, 12]:
            strong_beat_length //= beats_per_bar // 3
        # what about irregular time signatures? 5/x, 7/x

        return strong_beat_length

    def __append_messages(self, track: MidiTrack, messages: list[tuple]) -> None:
        # sort messages by start time, append them to track
        messages.sort()
        prev_abs_time = 0
        for message in messages:
            start, note, is_note_on, velocity, channel = message
            delta_time = start - prev_abs_time
            if is_note_on:
                track.append(
                    Message(
                        "note_on",
                        channel=channel,
                        note=note,
                        velocity=velocity,
                        time=int(delta_time),
                    )
                )
            else:
                track.append(
                    Message(
                        "note_off",
                        channel=channel,
                        note=note,
                        velocity=velocity,
                        time=int(delta_time),
                    )
                )
            prev_abs_time = message[0]

    def __filter_chords(
        self,
        chord_ppbs: Dict[tuple, float],
        prev_chord: tuple[int],
        melody_note: int,
        with_octave: bool,
        only_low_notes: bool,
    ) -> Dict[tuple, float]:
        if prev_chord is None:
            prev_chord = (melody_note,)

        if with_octave:  # long and ugly
            filtered_chord_ppbs = {
                chord: chord_ppbs[chord]
                for chord in filter(
                    lambda chord: (
                        not only_low_notes
                        or all(
                            [
                                utils.get_note_octave(note)
                                <= utils.HIGH_NOTES_OCTAVE_THRESHOLD
                                for note in chord
                            ]
                        )
                    )
                    and (
                        utils.get_note_name(melody_note)
                        in tuple(map(lambda note: utils.get_note_name(note), chord))
                        or len(set.intersection(set(chord), set(prev_chord))) > 1
                        or abs(chord[0] - prev_chord[0]) == 5  # perfect fourth
                        or abs(chord[0] - prev_chord[0]) == 7
                    ),  # perfect fifth
                    chord_ppbs.keys(),
                )
            }
        else:
            filtered_chord_ppbs = {
                chord: chord_ppbs[chord]
                for chord in filter(
                    lambda chord: utils.get_note_name(melody_note) in chord
                    or abs(
                        utils.NOTES.index(chord[0])
                        - utils.NOTES.index(utils.get_note_name(prev_chord[0]))
                    )
                    == 5  # perfect fourth
                    or abs(
                        utils.NOTES.index(chord[0])
                        - utils.NOTES.index(utils.get_note_name(prev_chord[0]))
                    )
                    == 7,  # perfect fifth
                    chord_ppbs.keys(),
                )
            }
        if not filtered_chord_ppbs:
            filtered_chord_ppbs = chord_ppbs
        return filtered_chord_ppbs

    def generate_music_with_melody_ngrams(
        self,
        output_file: str,
        bars: int,
        instrument_melody: int,
        instrument_harmony: int,
        melody_velocity: int = utils.DEFAULT_VELOCITY,
        harmony_velocity: int = utils.DEFAULT_VELOCITY,
        with_octave: bool = True,
        only_high_notes_melody: bool = False,
        only_low_notes_harmony: bool = False,
        first_note: str = None,
        tempo: int = None,
        lengths_flatten_factor: int = None,
        only_chords: bool = False,
        only_arpeggios: bool = False,
        more_chords: bool = False,
        # False - chord/arpeggio every start of bar,
        # True - every strong beat
        long_chords: bool = False,
        # False - chords/arpeggios of strong beat length
        # True - chords/arpeggios of chord_frequency length
    ) -> None:
        assert self.mm.fixed_time_signature
        assert not (only_chords and only_arpeggios)

        new_mid = MidiFile(
            type=1, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT
        )  # key, tempo and time signature in first track apply to all tracks
        track = self.__start_track(new_mid, instrument_melody, True)
        self.__set_tempo(track, tempo)
        self.__set_key(track)

        time_in_bar = 0
        bar_length = self.mm.main_bar_length
        beats_per_bar, _ = self.__set_time_signature(track)
        strong_beat_length = self.__calculate_strong_beat_length(
            bar_length, beats_per_bar
        )
        time_in_strong_beat = 0

        chord_track = self.__start_track(new_mid, instrument_harmony, False)
        if with_octave:
            all_count = sum(self.mm.chords.values())
            chords = self.mm.chords
        else:
            all_count = sum(self.mm.chords_without_octaves.values())
            chords = self.mm.chords_without_octaves
        chord_ppbs = {chord: count / all_count for chord, count in chords.items()}

        if not only_chords:
            arp_len = 4 if self.mm.simple_time else 3

        ret = self.__first_nminus1_tuples(
            with_octave, only_high_notes_melody, 0, True, first_note
        )
        if ret is None:
            raise ValueError(f"Can't start with note {first_note}!")
        prev_tuples, first_tuples = ret

        messages = (
            list()
        )  # list of tuples (absolute start time, note, if_note_on, velocity, channel)
        # ADDING MESSAGES LOOP
        prev_interval = 0
        prev_chord = None
        chord_frequency = bar_length if not more_chords else strong_beat_length
        chord_length = strong_beat_length if not long_chords else chord_frequency
        total_time = 0
        prev_note = -1
        while True:
            next_tuple, prev_tuples = self.__pick_tuple(
                first_tuples,
                prev_tuples,
                with_octave,
                only_high_notes_melody,
                prev_note,
                0,
                time_in_bar,
                bar_length,
            )
            next_note, next_note_length, next_interval = next_tuple
            if lengths_flatten_factor is not None:
                next_note_length, next_interval = self.mm.round_time(
                    next_note_length,
                    True,
                    self.mm.fixed_time_signature,
                    lengths_flatten_factor,
                ), self.mm.round_time(
                    next_interval,
                    False,
                    self.mm.fixed_time_signature,
                    lengths_flatten_factor,
                )

            offset = 0
            time_in_strong_beat = total_time % strong_beat_length
            time_in_bar = total_time % bar_length
            if (  # maybe extend interval
                time_in_strong_beat + next_note_length
                > strong_beat_length
                # and next_note_length <= strong_beat_length
                # and (time_in_strong_beat + next_note_length) % strong_beat_length != 0
            ):
                offset = (strong_beat_length - time_in_strong_beat) % strong_beat_length
                if time_in_bar + offset + next_note_length > bar_length:
                    offset = bar_length - time_in_bar
                prev_interval += offset
                time_in_strong_beat = 0
                time_in_bar = (time_in_bar + offset) % bar_length

            note_start_in_strong_beat = time_in_strong_beat
            time_in_strong_beat = (
                time_in_strong_beat + next_note_length
            ) % strong_beat_length

            if time_in_strong_beat == 0:  # maybe shrink interval
                next_interval = 0
            else:
                time_in_strong_beat += next_interval
                if (
                    time_in_strong_beat > strong_beat_length
                    and time_in_strong_beat % strong_beat_length != 0
                    # and next_interval <= strong_beat_length
                ):
                    next_interval -= time_in_strong_beat - strong_beat_length
                    time_in_strong_beat = 0

            total_time += offset
            if total_time // bar_length >= bars:
                break
            if time_in_bar == 0 or (more_chords and note_start_in_strong_beat == 0):
                iters = (
                    math.ceil((next_note_length + next_interval) / chord_frequency)
                    # 1
                    # if prev_interval + next_note_length < chord_frequency
                    # else (prev_interval + next_note_length) // chord_frequency
                )
                start_time = total_time
                for _ in range(iters):
                    filtered_chord_ppbs = self.__filter_chords(
                        chord_ppbs,
                        prev_chord,
                        next_note,
                        with_octave,
                        only_low_notes_harmony,
                    )
                    chords = list(filtered_chord_ppbs.keys())
                    ppbs = self.__normalize_ppbs(filtered_chord_ppbs)
                    # weighted random for now
                    chord = chords[np.random.choice(len(chords), p=ppbs)]
                    print(f"Chosen chord {chord}")

                    if not with_octave:
                        if prev_chord is None:
                            base_note = self.__specific_first_note(
                                chord[0], False, only_low_notes_harmony
                            )
                        else:
                            base_note = self.__next_closest_note(
                                prev_chord[0],
                                chord[0],
                                random.choice([True, False, None]),
                                False,
                                only_low_notes_harmony,
                            )
                        octave = utils.get_note_octave(base_note)
                        specific_chord = tuple()
                        for note in chord:
                            specific_note = utils.get_note_in_octave(note, octave)
                            while specific_note in specific_chord:
                                specific_note += 12
                            if utils.get_note_octave(
                                specific_note
                            ) > utils.HIGHEST_USED_OCTAVE or (
                                only_low_notes_harmony
                                and (
                                    utils.get_note_octave(specific_note)
                                    > utils.HIGH_NOTES_OCTAVE_THRESHOLD
                                )
                            ):
                                break
                            specific_chord += (specific_note,)
                        chord = tuple(set(specific_chord))

                    arpeggio = only_arpeggios
                    if not arpeggio:
                        if only_chords:
                            arpeggio = False
                        else:
                            arpeggio = random.choice([True, False])

                    if arpeggio:  # random arpeggios
                        while len(chord) < arp_len:
                            chord += (random.choice(chord),)
                        arp = random.choice(list(permutations(chord)))[:arp_len]

                        note_len = chord_length // len(arp)
                        for note in arp:
                            messages.append((start_time, note, True, harmony_velocity, 1))
                            messages.append((start_time + note_len, note, False, harmony_velocity, 1))
                            start_time += note_len
                        start_time = total_time
                    else:
                        for note in chord:
                            messages.append((start_time, note, True, harmony_velocity, 1))
                            messages.append((start_time + chord_length, note, False, harmony_velocity, 1))
                    prev_chord = chord
                    start_time += chord_length

            messages.append((total_time, next_note, True, melody_velocity, 0))
            messages.append((total_time + next_note_length, next_note, False, melody_velocity, 0))
            total_time += next_note_length + next_interval

            prev_note, prev_interval = next_note, next_interval

        melody_messages = list(filter(lambda msg: msg[4] == 0, messages))
        self.__append_messages(track, melody_messages)
        chord_messages = list(filter(lambda msg: msg[4] == 1, messages))
        self.__append_messages(chord_track, chord_messages)

        new_mid.save(os.path.join(os.path.dirname(__file__), output_file))
        self.__print_track(output_file)

    def __change_note_octave(self, note: int, chord: list[int], only_high_notes: bool):
        low_end = (
            utils.HIGH_NOTES_OCTAVE_THRESHOLD
            if only_high_notes
            else utils.LOWEST_USED_OCTAVE
        )
        notes_range = range(
            utils.get_note_in_octave("C", low_end),
            utils.get_note_in_octave("B", utils.HIGHEST_USED_OCTAVE) + 1,
        )

        # ugly, changes octave if the note is doubled
        jump = 12
        trials = 1
        # look in "neighbouring" octaves
        while note in chord or note not in notes_range:
            note += jump
            jump = -(jump + 12) if jump > 0 else -(jump - 12)
            trials += 1
            if trials == 2 * utils.OCTAVES:
                return None
        return note

    def generate_music_with_bar_ngrams(
        self,
        output_file: str,
        bars: int,
        instrument: int,
        velocity: int = utils.DEFAULT_VELOCITY,
        with_octave: bool = True,
        only_high_notes: bool = False,
        first_note: str = None,
        tempo: int = None,
        start_with_chord: bool = False,
    ) -> None:
        new_mid = MidiFile(
            type=0, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT
        )  # one track
        track = self.__start_track(new_mid, instrument, True)
        self.__set_tempo(track, tempo)
        self.__set_key(track)

        bar_length = self.mm.main_bar_length
        self.__set_time_signature(track)

        ret = self.__first_nminus1_tuples(
            with_octave, only_high_notes, 2, True, start_with_chord, first_note
        )
        if ret is None:
            raise ValueError(f"Can't start with note {first_note}!")
        prev_tuples, first_tuples = ret

        messages = list()  # list of tuples (absolute start time, note, if_note_on)
        # ADDING MESSAGES LOOP
        chord = set()
        total_time = 0
        prev_note, prev_time_in_bar = -1, -1
        while True:
            time_in_bar = total_time % bar_length
            next_tuple, prev_tuples = self.__pick_tuple(
                first_tuples,
                prev_tuples,
                with_octave,
                only_high_notes,
                prev_note,
                2,
                total_time % bar_length,
                bar_length,
                messages,
            )
            next_note, note_length, next_time_in_bar = next_tuple

            if not with_octave:
                # can return None
                next_note = self.__change_note_octave(next_note, chord, only_high_notes)

            if total_time // bar_length >= bars:
                break
            if next_note is not None and next_note not in chord:
                messages.append((total_time, next_note, True, velocity, 0))
                messages.append(
                    (total_time + note_length, next_note, False, velocity, 0)
                )
                prev_note = next_note
            else:
                continue

            if next_time_in_bar == prev_time_in_bar:
                chord.add(next_note)
            else:
                chord = set({next_note})

            if next_time_in_bar >= time_in_bar:
                total_time += next_time_in_bar - time_in_bar
            else:
                total_time += bar_length + next_time_in_bar - time_in_bar
            prev_time_in_bar = next_time_in_bar

        self.__append_messages(track, messages)

        new_mid.save(os.path.join(os.path.dirname(__file__), output_file))
        self.__print_track(output_file)

    def generate_music_with_tuple_ngrams(
        self,
        output_file: str,
        bars: int,
        instrument: int,
        velocity: int = utils.DEFAULT_VELOCITY,
        with_octave: bool = True,
        only_high_notes: bool = False,
        first_note: str = None,
        tempo: int = None,
        lengths_flatten_factor: int = None,
        start_with_chord: bool = False,
        strict_time_signature: bool = False,
    ) -> None:
        new_mid = MidiFile(
            type=0, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT
        )  # one track
        track = self.__start_track(new_mid, instrument, True)
        self.__set_tempo(track, tempo)
        self.__set_key(track)

        in_time_signature = self.mm.fixed_time_signature
        bar_length = self.mm.main_bar_length
        if in_time_signature:
            beats_per_bar, _ = self.__set_time_signature(track)
            strong_beat_length = self.__calculate_strong_beat_length(
                bar_length, beats_per_bar
            )
            time_in_strong_beat = 0

        ret = self.__first_nminus1_tuples(
            with_octave, only_high_notes, 1, True, start_with_chord, first_note
        )
        if ret is None:
            raise ValueError(f"Can't start with note {first_note}!")
        prev_tuples, first_tuples = ret

        messages = (
            list()
        )  # list of tuples (absolute start time, note, if_note_on, velocity, channel)
        # ADDING MESSAGES LOOP
        total_time = 0
        prev_note = -1
        chord = set()
        while True:
            next_tuple, prev_tuples = self.__pick_tuple(
                first_tuples,
                prev_tuples,
                with_octave,
                only_high_notes,
                prev_note,
                1,
                total_time % bar_length,
                bar_length,
                messages,
            )
            next_note, note_length, until_next_note_start = next_tuple
            if lengths_flatten_factor is not None:
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

            if not with_octave:
                # can return None
                next_note = self.__change_note_octave(next_note, chord, only_high_notes)

            if (
                until_next_note_start == 0 and len(chord) == utils.MAX_CHORD_SIZE - 1
            ) or next_note is None:
                # start new chord
                until_next_note_start = utils.UNTIL_NEXT_CHORD * note_length
                next_note = next_tuple[0]

            offset = 0
            if in_time_signature:
                time_in_strong_beat = total_time % strong_beat_length
                time_in_bar = total_time % bar_length
                if (  # maybe extend until_next_note_start
                    time_in_strong_beat + note_length
                    > strong_beat_length
                    # and next_note_length <= strong_beat_length
                    # and (time_in_strong_beat + next_note_length) % strong_beat_length != 0
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
                    # and until_next_note_start <= strong_beat_length
                ):
                    until_next_note_start -= time_in_strong_beat - strong_beat_length

            total_time += offset
            if total_time // bar_length >= bars:
                break
            if next_note not in chord:
                messages.append((total_time, next_note, True, velocity, 0))
                messages.append(
                    (total_time + note_length, next_note, False, velocity, 0)
                )
                prev_note = next_note
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

        self.__append_messages(track, messages)

        new_mid.save(os.path.join(os.path.dirname(__file__), output_file))
        self.__print_track(output_file)

    # very similar to previous method for now!
    def generate_music_from_file_nanogpt(
        self,
        input_filepath: str,
        output_file: str,
        instrument: int,
        velocity: int = utils.DEFAULT_VELOCITY,
        tempo: int = None,
        lengths_flatten_factor: int = None,
        strict_time_signature: bool = False,
    ) -> None:
        new_mid = MidiFile(
            type=0, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT
        )  # one track
        track = self.__start_track(new_mid, instrument, True)
        self.__set_tempo(track, tempo)
        self.__set_key(track)

        in_time_signature = self.mm.fixed_time_signature
        bar_length = self.mm.main_bar_length
        if in_time_signature:
            beats_per_bar, _ = self.__set_time_signature(track)
            strong_beat_length = self.__calculate_strong_beat_length(
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
        # for tuple in tuples:
        # next_note, note_length, until_next_note_start = map(int, tuple.split(","))
        while values:
            until_next_note_start, next_note, note_length = (
                int(values[0][1:]),
                int(values[1][1:]),
                int(values[2][1:]),
            )
            for i in range(3):
                values.pop(0)

            if lengths_flatten_factor is not None:
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
                    # and next_note_length <= strong_beat_length
                    # and (time_in_strong_beat + next_note_length) % strong_beat_length != 0
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
                    # and next_interval <= strong_beat_length
                ):
                    until_next_note_start -= time_in_strong_beat - strong_beat_length

            total_time += offset
            if next_note not in chord:
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
#     n=n,
#     dir=False,
#     pathname=pathname,
#     merge_tracks=True,
#     ignore_bass=False,
#     # key="C",
#     time_signature="3/4",
#     # lengths_flatten_factor=2,
# )

# or dirname - e.g. -d or --dir flag
pathname = "chopin"
mm = MarkovModel(
    n=n,
    dir=True,
    pathname=pathname,
    merge_tracks=True,
    ignore_bass=True,
    key="C",
    time_signature="3/4",
    # lengths_flatten_factor=2,
)

if mm.processed_mids == 0:
    raise ValueError("Couldn't process any mids! Try turning off key signature.")

generator = MusicGenerator(mm)
generator_uniform = MusicGenerator(mm, uniform=True)
generator_greedy = MusicGenerator(mm, k=1, weighted_random_start=True)
generator_k3 = MusicGenerator(mm, k=3)
generator_p80 = MusicGenerator(mm, p=0.8, weighted_random_start=True)

if __name__ == "__main__":
    # generator.generate_music_with_melody_ngrams(
    #     output_file="test1.mid",
    #     bars=40,
    #     instrument_melody=0,
    #     instrument_harmony=0,
    #     # melody_velocity=64,
    #     # harmony_velocity=35,
    #     with_octave=True,
    #     only_high_notes_melody=False,
    #     only_low_notes_harmony=False,
    #     # first_note="D",
    #     # tempo=80,
    #     lengths_flatten_factor=2,
    #     only_chords=True,
    #     # only_arpeggios=True,
    #     more_chords=False,
    #     long_chords=False,
    # )

    generator.generate_music_with_tuple_ngrams(
        output_file="test2.mid",
        bars=40,
        instrument=0,
        with_octave=True,
        only_high_notes=False,
        # first_note="G",
        tempo=80,
        lengths_flatten_factor=2,
        # start_with_chord=True,
        # velocity=80,
    )

    # generator.generate_music_with_bar_ngrams(
    #     output_file="test3.mid",
    #     bars=40,
    #     instrument=0,
    #     with_octave=True,
    #     only_high_notes=False,
    #     # tempo=80,
    #     # velocity=80,
    # )

    # # DIFFERENT SAMPLING METHODS
    # generator_uniform.generate_music_with_tuple_ngrams(
    #     output_file="test2_uniform.mid",
    #     bars=40,
    #     instrument=0,
    #     with_octave=True,
    #     only_high_notes=False,
    #     # first_note="C",
    #     # lengths_flatten_factor=2
    # )

    # generator_greedy.generate_music_with_tuple_ngrams(
    #     output_file="test2_greedy.mid",
    #     bars=20,
    #     instrument=0,
    #     with_octave=True,
    #     only_high_notes=False,
    #     first_note="D#",
    # )

    # generator_k3.generate_music_with_tuple_ngrams(
    #     output_file="test2_k3.mid",
    #     bars=40,
    #     instrument=0,
    #     with_octave=True,
    #     only_high_notes=False,
    #     # first_note="D",
    # )

    # generator_p80.generate_music_with_tuple_ngrams(
    #     output_file="test2_p80.mid",
    #     bars=40,
    #     instrument=0,
    #     with_octave=True,
    #     only_high_notes=False,
    #     # first_note="C",
    #     # lengths_flatten_factor=2
    # )

    # generator.generate_music_from_file_nanogpt(
    #     input_filepath="nanoGPT/test0.txt", output_file="test_gpt2.mid", instrument=0
    # )
