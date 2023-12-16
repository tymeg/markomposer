import os
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
import numpy as np
import random
from itertools import permutations
from typing import Union
from numpy.typing import NDArray

import utils
from markov import *


class MusicGenerator:
    def __init__(
        self,
        mm: MarkovModel,
        k: Optional[int] = None,
        p: Optional[float] = None,
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
        ppbs: Dict[Tuple, float],
        ngrams: Dict[Tuple, int],
        nminus1gram_count: int,
        prev: Tuple,
        value: Tuple,
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

    def __normalize_ppbs(self, ppbs: Dict[Tuple, float]) -> NDArray[np.float64]:
        # make list only of ppbs
        ppbs = list(ppbs.values())

        # normalize
        ppbs = np.array(ppbs, dtype="float64")
        ppbs /= ppbs.sum()
        return ppbs

    def __sampling(self, ppbs: Dict[Tuple, float]) -> None:
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
        self, note: Union[int, str], with_octave: bool, only_high_notes: bool
    ) -> bool:
        if with_octave and only_high_notes:
            if utils.get_note_octave(note) < utils.HIGH_NOTES_OCTAVE_THRESHOLD:
                return False
        return True

    def __choose_next_tuple(
        self,
        prev_tuples: Tuple[Union[Tuple[int], Tuple[int, bool]]],
        with_octave: bool,
        only_high_notes: bool,
        type: int,
    ) -> Optional[Tuple[int]]:
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
                ngrams = self.mm.harmony_ngrams
                nminus1grams = self.mm.harmony_nminus1grams
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
                ngrams = self.mm.harmony_ngrams_without_octaves
                nminus1grams = self.mm.harmony_nminus1grams_without_octaves
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
        start_of_bar: bool = False,
        first_note: Optional[Union[int, str]] = None,
    ) -> Optional[Tuple[Tuple[Tuple[Union[int, str]]], List[Tuple[Union[int, str]]]]]:
        if with_octave:
            if type == 0:
                nminus1grams = self.mm.melody_nminus1grams
                starts_of_bar = self.mm.melody_nminus1gram_starts_of_bar
            elif type == 1:
                nminus1grams = self.mm.harmony_nminus1grams
                starts_of_bar = self.mm.harmony_nminus1gram_starts_of_bar
            elif type == 2:
                nminus1grams = self.mm.bar_nminus1grams
        else:
            if type == 0:
                nminus1grams = self.mm.melody_nminus1grams_without_octaves
                starts_of_bar = self.mm.melody_nminus1gram_without_octaves_starts_of_bar
            elif type == 1:
                nminus1grams = self.mm.harmony_nminus1grams_without_octaves
                starts_of_bar = (
                    self.mm.harmony_nminus1gram_without_octaves_starts_of_bar
                )
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
        first_tuples: List[Tuple[int]],
        prev_tuples: Tuple[Tuple[int]],
        with_octave: bool,
        only_high_notes: bool,
        prev_note: int,
        type: int,
        time_in_bar: int,
        bar_length: int,
        messages: Optional[List[Tuple[int, bool]]] = None,
    ) -> Tuple[Tuple[int], Tuple[Tuple[int]]]:
        if first_tuples:
            next_tuple = first_tuples.pop(0)
            # print(f"Chosen {next_tuple}")
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
                # print(f"Chosen {next_tuple}")
            else:
                # print(f"Chosen {next_tuple} after {prev_tuples}")
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
    def _start_track(
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

    def _set_tempo(self, track: MidiTrack, tempo: Optional[int]) -> None:
        if tempo is not None:
            tempo = bpm2tempo(tempo)
        else:
            tempo = self.mm.main_tempo
        track.append(MetaMessage("set_tempo", tempo=tempo))

    def _set_key(self, track: MidiTrack) -> None:
        key = self.mm.main_key
        if key is not None:
            track.append(MetaMessage("key_signature", key=key))

    def _set_time_signature(self, track: MidiTrack) -> Tuple[int]:
        beats_per_bar, beat_value = self.mm.main_beats_per_bar, self.mm.main_beat_value
        track.append(
            MetaMessage(
                "time_signature",
                numerator=beats_per_bar,
                denominator=beat_value,
            )
        )

        return beats_per_bar, beat_value

    def _print_track(self, output_file: str) -> None:
        print("Generated track:")
        test_mid = MidiFile(os.path.join(os.path.dirname(__file__), output_file))
        for track_idx, track in enumerate(test_mid.tracks):
            print(f"Track {track_idx}: {track.name}")
            for msg in track:
                print(msg)

    # ========================= MAIN GENERATING METHODS =================================================
    def _calculate_strong_beat_length(self, bar_length: int, beats_per_bar: int) -> int:
        strong_beat_length = bar_length
        # simple time signatures
        if beats_per_bar in [2, 3, 4]:
            strong_beat_length //= beats_per_bar // 2
        # compound time signatures
        elif beats_per_bar in [6, 9, 12]:
            strong_beat_length //= beats_per_bar // 3
        # what about irregular time signatures? 5/x, 7/x

        return strong_beat_length

    def _add_tonic_chord(
        self,
        prev_chord: Set[int],
        messages: List[Tuple],
        end_of_chord: bool,
        total_time: int,
        bar_length: int,
        velocity: int,
        channel: int,
    ) -> None:
        tonic_note = utils.get_tonic_note(self.mm.main_key)
        prev_base_note = sorted(list(prev_chord))[0]
        octave = utils.get_note_octave(prev_base_note)
        base_note = utils.get_note_in_octave(tonic_note, octave)
        chord = (
            [base_note, base_note + 3, base_note + 7]
            if utils.is_minor(self.mm.main_key)
            else [base_note, base_note + 4, base_note + 7]
        )
        if not (end_of_chord and chord == prev_chord):
            for note in chord:
                messages.append((total_time, note, True, velocity, channel))
                messages.append(
                    (total_time + bar_length, note, False, velocity, channel)
                )

    def _append_messages(self, track: MidiTrack, messages: List[Tuple]) -> None:
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
        chord_ppbs: Dict[Tuple, float],
        melody_notes: Set[str],
        with_octave: bool,
        only_low_notes: bool,
    ) -> Dict[Tuple, float]:
        use_simple_chord = True
        if len(melody_notes) > 1:
            use_simple_chord = False
        if with_octave:  # long and ugly
            max_melody_match = max(
                chord_ppbs.keys(),
                key=lambda chord: len(
                    set.intersection(
                        set(map(lambda note: utils.get_note_name(note), chord)),
                        melody_notes,
                    )
                ),
            )

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
                        (not use_simple_chord or utils.is_simple(chord, False))
                        and len(
                            set.intersection(
                                set(map(lambda note: utils.get_note_name(note), chord)),
                                melody_notes,
                            )
                        )
                        == max_melody_match
                    ),
                    chord_ppbs.keys(),
                )
            }
        else:
            max_melody_match = max(
                chord_ppbs.keys(),
                key=lambda chord: len(set.intersection(set(chord), melody_notes)),
            )
            filtered_chord_ppbs = {
                chord: chord_ppbs[chord]
                for chord in filter(
                    lambda chord: (not use_simple_chord or utils.is_simple(chord, True))
                    and len(set.intersection(set(chord), melody_notes))
                    == max_melody_match,
                    chord_ppbs.keys(),
                )
            }
        if not filtered_chord_ppbs:
            filtered_chord_ppbs = chord_ppbs
        return filtered_chord_ppbs

    # METHOD 1
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
        first_note: Optional[str] = None,
        tempo: Optional[int] = None,
        lengths_flatten_factor: Optional[int] = None,
        only_chords: bool = False,
        only_arpeggios: bool = False,
        more_chords: bool = False,
        # False - chord/arpeggio every start of bar,
        # True - every strong beat
        long_chords: bool = False,
        # False - chords/arpeggios of strong beat length
        # True - chords/arpeggios of chord_frequency length
        end_on_tonic: bool = False,
    ) -> None:
        new_mid = MidiFile(
            type=1, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT
        )  # key, tempo and time signature in first track apply to all tracks
        track = self._start_track(new_mid, instrument_melody, True)
        self._set_tempo(track, tempo)
        self._set_key(track)

        time_in_bar = 0
        bar_length = self.mm.main_bar_length
        beats_per_bar, _ = self._set_time_signature(track)
        strong_beat_length = self._calculate_strong_beat_length(
            bar_length, beats_per_bar
        )
        time_in_strong_beat = 0

        chord_track = self._start_track(new_mid, instrument_harmony, False)
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
        melody_notes = set()
        total_time, start_time = 0, 0
        prev_note = -1
        progress = tqdm(total=bars)
        bar = 0
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
            next_note_length, next_interval = self.mm.round_time(
                next_note_length,
                True,
                True,
                lengths_flatten_factor,
            ), self.mm.round_time(
                next_interval,
                False,
                True,
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
            # harmonization
            if melody_notes and (
                time_in_bar == 0 or (more_chords and note_start_in_strong_beat == 0)
            ):
                iters = (total_time - start_time) // chord_frequency
                for _ in range(iters):
                    filtered_chord_ppbs = self.__filter_chords(
                        chord_ppbs,
                        melody_notes,
                        with_octave,
                        only_low_notes_harmony,
                    )
                    chords = list(filtered_chord_ppbs.keys())
                    ppbs = self.__normalize_ppbs(filtered_chord_ppbs)
                    # weighted random for now
                    chord = chords[np.random.choice(len(chords), p=ppbs)]
                    # print(f"Chosen chord {chord}")

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
                            messages.append(
                                (start_time, note, True, harmony_velocity, 1)
                            )
                            messages.append(
                                (
                                    start_time + note_len,
                                    note,
                                    False,
                                    harmony_velocity,
                                    1,
                                )
                            )
                            start_time += note_len
                        start_time = total_time
                    else:
                        for note in chord:
                            messages.append(
                                (start_time, note, True, harmony_velocity, 1)
                            )
                            messages.append(
                                (
                                    start_time + chord_length,
                                    note,
                                    False,
                                    harmony_velocity,
                                    1,
                                )
                            )
                    prev_chord = chord
                    start_time += chord_length
                melody_notes = set()

            if total_time // bar_length > bar:
                bar += 1
                progress.update()
            if total_time // bar_length >= bars:
                if end_on_tonic:
                    if only_arpeggios:
                        tonic_note = utils.get_tonic_note(self.mm.main_key)
                        prev_base_note = sorted(list(prev_chord))[0]
                        octave = utils.get_note_octave(prev_base_note)
                        tonic_note = utils.get_note_in_octave(tonic_note, octave)
                        messages.append(
                            (total_time, tonic_note, True, harmony_velocity, 1)
                        )
                        messages.append(
                            (
                                total_time + bar_length,
                                tonic_note,
                                False,
                                harmony_velocity,
                                1,
                            )
                        )
                    else:
                        self._add_tonic_chord(
                            prev_chord,
                            messages,
                            False,
                            total_time,
                            bar_length,
                            harmony_velocity,
                            1,
                        )
                break

            melody_notes.add(utils.get_note_name(next_note))
            messages.append((total_time, next_note, True, melody_velocity, 0))
            messages.append(
                (total_time + next_note_length, next_note, False, melody_velocity, 0)
            )
            total_time += next_note_length + next_interval

            prev_note, prev_interval = next_note, next_interval

        melody_messages = list(filter(lambda msg: msg[4] == 0, messages))
        self._append_messages(track, melody_messages)
        chord_messages = list(filter(lambda msg: msg[4] == 1, messages))
        self._append_messages(chord_track, chord_messages)

        new_mid.save(os.path.join(os.getcwd(), output_file))
        # self.__print_track(output_file)

    def __change_note_octave(
        self, note: int, chord: List[int], only_high_notes: bool
    ) -> Optional[int]:
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

    # for broad_chords
    def __good_note(self, note: int, chord: Set[int]) -> bool:
        for chord_note in chord:
            if abs(note - chord_note) < 3:
                return False
        return True

    # METHOD 2
    def generate_music_with_harmony_ngrams(
        self,
        output_file: str,
        bars: int,
        instrument: int,
        velocity: int = utils.DEFAULT_VELOCITY,
        with_octave: bool = True,
        only_high_notes: bool = False,
        first_note: Optional[str] = None,
        tempo: Optional[int] = None,
        lengths_flatten_factor: Optional[int] = None,
        strict_time_signature: bool = False,
        start_filepath: Optional[str] = None,
        end_on_tonic: bool = False,
        broad_chords: bool = False,
    ) -> None:
        new_mid = MidiFile(
            type=0, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT
        )  # one track
        track = self._start_track(new_mid, instrument, True)
        self._set_tempo(track, tempo)
        self._set_key(track)

        in_time_signature = self.mm.fixed_time_signature
        bar_length = self.mm.main_bar_length
        if in_time_signature:
            beats_per_bar, _ = self._set_time_signature(track)
            strong_beat_length = self._calculate_strong_beat_length(
                bar_length, beats_per_bar
            )
            time_in_strong_beat = 0

        if start_filepath:
            with open(start_filepath, "r") as f:
                first_tuples = list()
                values = f.read().split()
                values.pop(0)  # skip START token
                while values:
                    first_tuples.append(
                        (int(values[1][1:]), int(values[2][1:]), int(values[0][1:]))
                    )
                    for _ in range(3):
                        values.pop(0)
                prev_tuples = tuple(first_tuples[-(self.mm.n - 1) :])
        else:
            ret = self.__first_nminus1_tuples(
                with_octave, only_high_notes, 1, True, first_note
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
        prev_chord = None
        progress = tqdm(total=bars)
        bar = 0
        while True:
            if len(first_tuples) > self.mm.n - 1:
                next_tuple = first_tuples.pop(0)
            else:
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
            if total_time // bar_length > bar:
                bar += 1
                progress.update()
            if total_time // bar_length >= bars:
                if end_on_tonic:
                    self._add_tonic_chord(
                        prev_chord,
                        messages,
                        end_of_chord,
                        total_time,
                        bar_length,
                        velocity,
                        0,
                    )
                break
            if next_note not in chord and (
                not broad_chords or self.__good_note(next_note, chord)
            ):
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
                    prev_chord = chord
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

        self._append_messages(track, messages)

        new_mid.save(os.path.join(os.getcwd(), output_file))
        # self.__print_track(output_file)

    # METHOD 3
    def generate_music_with_bar_ngrams(
        self,
        output_file: str,
        bars: int,
        instrument: int,
        velocity: int = utils.DEFAULT_VELOCITY,
        with_octave: bool = True,
        only_high_notes: bool = False,
        first_note: Optional[str] = None,
        tempo: Optional[int] = None,
        end_on_tonic: bool = False,
        broad_chords: bool = False,
    ) -> None:
        new_mid = MidiFile(
            type=0, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT
        )  # one track
        track = self._start_track(new_mid, instrument, True)
        self._set_tempo(track, tempo)
        self._set_key(track)

        bar_length = self.mm.main_bar_length
        self._set_time_signature(track)

        ret = self.__first_nminus1_tuples(
            with_octave, only_high_notes, 2, True, first_note
        )
        if ret is None:
            raise ValueError(f"Can't start with note {first_note}!")
        prev_tuples, first_tuples = ret

        messages = (
            list()
        )  # list of tuples (absolute start time, note, if_note_on, velocity, channel)
        # ADDING MESSAGES LOOP
        chord = set()
        total_time = 0
        prev_note, prev_time_in_bar = -1, -1
        prev_chord = None
        progress = tqdm(total=bars)
        bar = 0
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

            if total_time // bar_length > bar:
                bar += 1
                progress.update()
            if total_time // bar_length >= bars:
                if end_on_tonic:
                    self._add_tonic_chord(
                        prev_chord,
                        messages,
                        end_of_chord,
                        total_time,
                        bar_length,
                        velocity,
                        0,
                    )
                break
            if next_note is not None and (
                len(chord) == 1
                or (
                    next_note not in chord
                    and (not broad_chords or self.__good_note(next_note, chord))
                )
            ):
                messages.append((total_time, next_note, True, velocity, 0))
                messages.append(
                    (total_time + note_length, next_note, False, velocity, 0)
                )
                prev_note = next_note
            else:
                continue

            end_of_chord = False
            if next_time_in_bar == prev_time_in_bar:
                chord.add(next_note)
            else:
                if len(chord) > 1:
                    end_of_chord = True
                    prev_chord = chord
                chord = set({next_note})

            if next_time_in_bar >= time_in_bar:
                total_time += next_time_in_bar - time_in_bar
            else:
                total_time += bar_length + next_time_in_bar - time_in_bar
            prev_time_in_bar = next_time_in_bar

        self._append_messages(track, messages)

        new_mid.save(os.path.join(os.getcwd(), output_file))
        # self.__print_track(output_file)
