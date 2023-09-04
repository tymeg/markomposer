import os
import utils
from mido import MidiFile, MetaMessage, tempo2bpm
import math


class MarkovModel:
    def __init__(self, n: int, m: int, filepath: str) -> None:
        self.n = n  # note n-grams
        self.m = m  # length m-grams
        self.filepath = filepath

        self.mid = MidiFile(os.path.join(os.path.dirname(__file__), filepath))
        print(self.mid.ticks_per_beat)

        self.ticks_per_beat_factor = (
            utils.DEFAULT_TICKS_PER_BEAT / self.mid.ticks_per_beat
        )

        print(f"Track's length [sec]: {self.mid.length}")
        print(f"File type: {self.mid.type}")
        if self.mid.type == 2:  # maybe we should handle them in the future
            raise ValueError(".mid file should be of type 0 or 1!")

        # NOTES
        # n-grams
        self.note_ngrams = {}  # numbers -> how many
        self.note_ngrams_without_octaves = {}  # strings -> how many
        # n-1-grams
        self.note_nminus1grams = {}  # numbers -> how many
        self.note_nminus1grams_without_octaves = {}  # strings -> how many

        # default for beat = quarter note, changed in process_midi if beat value is different
        self.length_precision = utils.DEFAULT_TICKS_PER_BEAT // 8

        # NOTE LENGTHS
        self.note_length_ngrams = {}
        self.note_length_nminus1grams = {}
        self.note_length_counts = {}

        # INTERVALS
        self.interval_ngrams = {}
        self.interval_nminus1grams = {}
        self.interval_counts = {}

        # main is currently first, maybe it should be the longest?
        # but main_tempo is average tempo
        self.main_beats_per_bar, self.main_beat_value = 0, 0
        self.main_key = ""
        self.main_tempo, self.tempos_count = 0, 0

        self.__process_midi(self.mid)

    def __process_midi(self, mid: MidiFile) -> None:
        for track_idx, track in enumerate(mid.tracks):
            total_time = 0
            notes = []

            # list of tuples (start of note, note length) to sort by start
            note_lengths = []
            # times in ticks between starts of consecutive notes
            intervals = []
            interval = 0
            currently_playing_notes_starts = {}

            print(f"Track {track_idx}: {track.name}")
            for msg in track:
                # print(msg)
                total_time += msg.time
                interval += msg.time

                if msg.is_meta:
                    self.__read_meta_message(msg)
                else:
                    if msg.type == "note_on" and msg.velocity > 0:
                        currently_playing_notes_starts[msg.note] = total_time
                        # round up
                        intervals.append(
                            # better round up or down? I want 0 length intervals
                            math.floor(
                                interval
                                * self.ticks_per_beat_factor
                                / self.length_precision
                            )
                            * self.length_precision
                        )
                        notes.append(msg.note)
                        # print(utils.get_note_name_with_octave(msg.note))
                        # print(total_time)
                    if (
                        msg.type == "note_on" and msg.velocity == 0
                    ) or msg.type == "note_off":
                        start = currently_playing_notes_starts[msg.note]
                        # round up
                        note_lengths.append(
                            (
                                # better round up or down? I don't want 0 length notes
                                start,
                                math.ceil(
                                    (total_time - start)
                                    * self.ticks_per_beat_factor
                                    / self.length_precision
                                )
                                * self.length_precision,
                            )
                        )
                        interval = 0

            print(f"Track {track_idx} notes: \n{notes}")
            note_lengths.sort()
            note_lengths = list(map(lambda tpl: tpl[1], note_lengths))
            print(f"Track {track_idx} note lengths: \n{note_lengths}")
            print(f"Track {track_idx} intervals: \n{intervals}")

            if notes:
                self.__count_track_note_ngrams(notes)

                self.__count_track_length_ngrams(note_lengths, True)
                self.__count_track_length_ngrams(intervals, False)

                self.__count_track_length_occurences(note_lengths, True)
                self.__count_track_length_occurences(intervals, False)

        if self.tempos_count > 1:
            self.main_tempo = self.main_tempo // self.tempos_count  # average

        print(f"\nNotes n-grams: \n{self.note_ngrams}\n")
        print(f"Notes n-grams without octaves: \n{self.note_ngrams_without_octaves}\n")
        print(f"\nNotes n-1-grams: \n{self.note_nminus1grams}\n")
        print(
            f"Notes n-1-grams without octaves: \n{self.note_nminus1grams_without_octaves}\n"
        )

        print(f"Note length counts: \n{self.note_length_counts}")
        print(f"Interval counts: \n{self.interval_counts}")

    def __read_meta_message(self, msg: MetaMessage) -> None:
        # TODO: do I need current tempo, key etc. for anything?
        if msg.type == "set_tempo":
            current_tempo = msg.tempo
            self.main_tempo += current_tempo
            self.tempos_count += 1
            # print(
            #     f"Tempo: {tempo2bpm(current_tempo)} BPM ({current_tempo} microseconds per quarter note)"
            # )
        if msg.type == "time_signature":
            current_beats_per_bar = msg.numerator
            if self.main_beats_per_bar == 0:
                self.main_beats_per_bar = current_beats_per_bar
            current_beat_value = msg.denominator
            if self.main_beat_value == 0:
                self.main_beat_value = current_beat_value
                # good?
                self.length_precision = utils.DEFAULT_TICKS_PER_BEAT // (
                    32 // self.main_beat_value
                )
            print(f"Time signature: {current_beats_per_bar}/{current_beat_value}")
        if msg.type == "key_signature":
            current_key = msg.key
            if self.main_key == "":
                self.main_key = current_key
            print(f"Key: {current_key}")

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

            if self.note_nminus1grams.get(nminus1gram) is not None:
                self.note_nminus1grams[nminus1gram] += 1
                self.note_nminus1grams_without_octaves[nminus1gram_without_octaves] += 1
            else:
                self.note_nminus1grams[nminus1gram] = 1
                if (
                    self.note_nminus1grams_without_octaves.get(
                        nminus1gram_without_octaves
                    )
                    is not None
                ):
                    self.note_nminus1grams_without_octaves[
                        nminus1gram_without_octaves
                    ] += 1
                else:
                    self.note_nminus1grams_without_octaves[
                        nminus1gram_without_octaves
                    ] = 1

            # not last n-1 notes -> count n-gram
            if note_idx != len(notes) - self.n + 1:
                ngram = nminus1gram + (notes[note_idx + self.n - 1],)
                ngram_without_octaves = nminus1gram_without_octaves + (
                    utils.get_note_name(notes[note_idx + self.n - 1]),
                )

                if self.note_ngrams.get(ngram) is not None:
                    self.note_ngrams[ngram] += 1
                    self.note_ngrams_without_octaves[ngram_without_octaves] += 1
                else:
                    self.note_ngrams[ngram] = 1
                    if (
                        self.note_ngrams_without_octaves.get(ngram_without_octaves)
                        is not None
                    ):
                        self.note_ngrams_without_octaves[ngram_without_octaves] += 1
                    else:
                        self.note_ngrams_without_octaves[ngram_without_octaves] = 1

    def __count_track_length_ngrams(
        self, lengths: list[int], if_note_lengths: bool
    ) -> None:
        if if_note_lengths:
            ngrams = self.note_length_ngrams
            nminus1grams = self.note_length_nminus1grams
        else:
            ngrams = self.interval_ngrams
            nminus1grams = self.interval_nminus1grams

        for length_idx in range(len(lengths) - self.m + 2):
            # count n-1-gram
            nminus1gram = tuple(
                [lengths[length_idx + offset] for offset in range(self.m - 1)]
            )

            if nminus1grams.get(nminus1gram) is not None:
                nminus1grams[nminus1gram] += 1
            else:
                nminus1grams[nminus1gram] = 1

            # not last n-1 notes -> count n-gram
            if length_idx != len(lengths) - self.m + 1:
                ngram = nminus1gram + (lengths[length_idx + self.m - 1],)

                if ngrams.get(ngram) is not None:
                    ngrams[ngram] += 1
                else:
                    ngrams[ngram] = 1

    def __count_track_length_occurences(
        self, lengths: list[int], if_note_lengths: bool
    ) -> None:
        max_length = max(lengths)

        if if_note_lengths:
            counts = self.note_length_counts
        else:
            counts = self.interval_counts

        for length in range(0, max_length + 1, self.length_precision):
            if counts.get(length) is None:
                counts[length] = lengths.count(length)
            else:
                counts[length] += lengths.count(length)
