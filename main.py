import argparse
from mido import second2tick
import contextlib
with contextlib.redirect_stdout(None):
    import pygame

from gen_music import *

parser = argparse.ArgumentParser(
    description="Generate music in style of input .mid files, using Markov chains",
    formatter_class=argparse.RawTextHelpFormatter,
)

# -------------------------------- REQUIRED ARGUMENTS ------------------------------------
parser.add_argument(
    "input_path", help="path to input .mid file or directory with .mid files"
)
parser.add_argument(
    "method",
    choices=[1, 2, 3],
    type=int,
    help="generating method:\n"
    '1 - "melody n-grams" - melody from Markov chain, later harmonized with chords/arpeggios (2 tracks)\n'
    '2 - "harmony n-grams" - melody and harmony from Markov chain (notes with distances between them) (1 track)\n'
    '3 - "bar n-grams" - melody and harmony from Markov chain (notes with positions in bar) (1 track)\n\n'
    "Best use 3 when most/all mids are in the same time signature (and have multiple instruments, e.g. pop music)\n"
    "2 is better for single instrument mids in different time signatures (e.g. classical piano music)\n"
    "You can always use 1, but it can generate music a bit less similar to input mids than methods 2 and 3.",
)
parser.add_argument("length", help="time in bars (int) or minutes and seconds - m:s\n"\
                    "In case of using --start-filepath, start notes are counted in the length!")
parser.add_argument("output_filename", help="output file name", default="music.mid")

# ----------------------------------- OPTIONALS -----------------------------------------
parser.add_argument("-n", "--n", type=int, help="n-gram length (default: 3)", default=3)

# time signature
beats_per_bar = {2, 3, 4, 6, 9, 12}
beat_values = {2, 4, 8, 16, 32}
parser.add_argument(
    "-ts",
    "--time-signature",
    help=f"b/v where b in {beats_per_bar} and v in {beat_values}\n"
    "Default: doesn't force any time signature.\n"
    "Note: can work imperfectly. Also, when using 3. generating method,\n"
    "specify time signature common for most input files!",
)

# key signature
parser.add_argument(
    "-k",
    "--key",
    choices=utils.KEY_SIGNATURES,
    help="Default: doesn't force any key signature.\n"
    "Note: this doesn't work strictly - notes outside of scale can appear.\n"
    "Also, can be quirky if input mids haven't got key encoded/have multiple keys/wrong key encoding\n"
    "or are neither major nor minor.",
)

# tempo
parser.add_argument(
    "-t",
    "--tempo",
    type=int,
    help="tempo in BPM\nDefault: average tempo from input mids.\n"
    'Note: this is "relative" tempo - tempo can vary because of \n'
    "different tempos and time signatures in input mids\n"
    "To make it less chaotic consider using --flatten-before or --flatten after flags",
)

# first note
parser.add_argument("-fn", "--first_note", choices=utils.NOTES)

# without octaves
parser.add_argument(
    "-wo",
    "--without-octaves",
    action="store_true",
    help="Markov chain works on 12 symbolic notes, rather than on notes in specific octaves",
    default=False,
)

# merge tracks, ignore bass
parser.add_argument(
    "-nm", "--no-merge", action="store_true", help="don't merge tracks", default=False
)
parser.add_argument(
    "-ib",
    "--ignore-bass",
    action="store_true",
    help="ignore bass tracks",
    default=False,
)

# sampling
sampling = parser.add_mutually_exclusive_group()
sampling.add_argument(
    "-u",
    "--uniform",
    action="store_true",
    help="sampling nth note from uniform distribution",
    default=False,
)
sampling.add_argument("--top_k", type=int, help="k for top-k sampling, k is int > 0")
sampling.add_argument("--top_p", type=float, help="p for top-p sampling, p from [0; 1]")
parser.add_argument(
    "-w",
    "--weighted-random-start",
    action="store_true",
    help="choose first n-1 notes by weighted random sampling",
    default=False,
)

# end on tonic note
parser.add_argument(
    "-e",
    "--end-on-tonic",
    action="store_true",
    help="end generated music on tonic note (resolving tension).\n"
    "Note: can make music a bit longer than specified.",
    default=False,
)

# ----------------------------- METHOD SPECIFIC OPTIONALS ----------------------------------

# METHOD 1
melody_ngrams_optionals = parser.add_argument_group("flags for method 1")

# instruments
melody_ngrams_optionals.add_argument(
    "-im",
    "--instrument-melody",
    type=int,
    help="instrument number from 0 to 127 playing melody (default: 0 - acoustic grand piano)",
    default=0,
)
melody_ngrams_optionals.add_argument(
    "-ih",
    "--instrument-harmony",
    type=int,
    help="instrument number from 0 to 127 playing harmony (default: 0 - acoustic grand piano)",
    default=0,
)

# velocities
melody_ngrams_optionals.add_argument(
    "-mv",
    "--melody-velocity",
    type=int,
    help=f"melody loudness from 0 to 127 (default: {utils.DEFAULT_VELOCITY})",
    default=utils.DEFAULT_VELOCITY,
)
melody_ngrams_optionals.add_argument(
    "-hv",
    "--harmony-velocity",
    type=int,
    help=f"harmony loudness from 0 to 127 (default: {utils.DEFAULT_VELOCITY})",
    default=utils.DEFAULT_VELOCITY,
)

# only high notes melody, only low notes harmony
melody_ngrams_optionals.add_argument(
    "-oh",
    "--only-high-notes-melody",
    action="store_true",
    help="melody only played on high octaves' notes",
    default=False,
)
melody_ngrams_optionals.add_argument(
    "-ol",
    "--only-low-notes_harmony",
    action="store_true",
    help="harmony only played on low octaves' notes\n"
    "Recommended usage: for bass instruments, used together with --only_arpeggios flag",
    default=False,
)

# only chords/arpeggios
melody_ngrams_optionals.add_argument(
    "-oc",
    "--only-chords",
    action="store_true",
    help="only chords in the harmony track",
    default=False,
)
melody_ngrams_optionals.add_argument(
    "-oa",
    "--only-arpeggios",
    action="store_true",
    help="only arpeggios in the harmony track",
    default=False,
)

# more/long chords
melody_ngrams_optionals.add_argument(
    "-mc",
    "--more-chords",
    action="store_true",
    help="chords/arpeggios every strong beat",
    default=False,
)
melody_ngrams_optionals.add_argument(
    "-lc",
    "--long-chords",
    action="store_true",
    help="chords/arpeggios play without pauses",
    default=False,
)

# METHOD 2
harmony_ngrams_optionals = parser.add_argument_group("flags for method 2")

# start filepath
harmony_ngrams_optionals.add_argument(
    "-sf",
    "--start-filepath",
    help="path to file with start notes in nanoGPT format, e.g. I0 N60 L120. Generator will continue the track.",
)

# strict time signature
harmony_ngrams_optionals.add_argument(
    "-st",
    "--strict-time-signature",
    action="store_true",
    help="longer notes can only play on start of bar",
    default=False,
)

# METHOD 2 AND 3
harmony_and_bar_ngrams_optionals = parser.add_argument_group("flags for method 2 and 3")

# instrument
harmony_and_bar_ngrams_optionals.add_argument(
    "-i",
    "--instrument",
    type=int,
    help="instrument number from 0 to 127 (default: 0 - acoustic grand piano)",
    default=0,
)

# velocity
harmony_and_bar_ngrams_optionals.add_argument(
    "-v",
    "--velocity",
    type=int,
    help=f"melody loudness from 0 to 127 (default: {utils.DEFAULT_VELOCITY})",
    default=utils.DEFAULT_VELOCITY,
)

# only high notes
harmony_and_bar_ngrams_optionals.add_argument(
    "-o",
    "--only-high-notes",
    action="store_true",
    help="track played only on high octaves' notes",
    default=False,
)

# start with chord
harmony_and_bar_ngrams_optionals.add_argument(
    "-sc",
    "--start-with-chord",
    action="store_true",
    help="generated track starts with a chord",
    default=False,
)

# flattens
harmony_and_bar_ngrams_optionals.add_argument(
    "-fb",
    "--flatten-before",
    type=int,
    choices=[2, 4, 8],
    help='factor by which 32th note length precision is multiplied before creating model. It "unifies" tempo.\n'\
    'Note: can "glue" different lengths with each other in the model.\n'\
    'Bonus feature: generation is faster.',
)
harmony_and_bar_ngrams_optionals.add_argument(
    "-fa",
    "--flatten-after",
    type=int,
    choices=[2, 4, 8],
    help='factor by which 32th note length precision is multiplied after creating model, during generation. It "unifies" tempo.',
)

# --------------------------------------- PARSING -------------------------------------
args = parser.parse_args()

if args.n < 2:
    raise ValueError("n must be >= 2!")
dir = not (args.input_path[-4:] == ".mid")
merge_tracks = not args.no_merge
with_octave = not args.without_octaves

mm = MarkovModel(
    n=args.n,
    dir=dir,
    pathname=args.input_path,
    merge_tracks=merge_tracks,
    ignore_bass=args.ignore_bass,
    key=args.key,
    time_signature=args.time_signature,
    lengths_flatten_factor=args.flatten_before,
)

if mm.processed_mids == 0:
    raise ValueError("Couldn't process any mids! Try turning off key signature.")

generator = MusicGenerator(
    mm,
    k=args.top_k,
    p=args.top_p,
    uniform=args.uniform,
    weighted_random_start=args.weighted_random_start,
)

output_file = args.output_filename
if args.output_filename[-4:] != ".mid":
    output_file += ".mid"

if args.tempo:
    tempo = bpm2tempo(args.tempo)
else:
    tempo = mm.main_tempo

if ":" in args.length:
    minutes, seconds = list(map(int, args.length.split(":")))
    seconds += 60 * minutes
    ticks = second2tick(seconds, utils.DEFAULT_TICKS_PER_BEAT, tempo)

    time_signature = args.time_signature
    if not time_signature:
        ticks_per_bar = utils.DEFAULT_BEATS_PER_BAR * utils.DEFAULT_TICKS_PER_BEAT
    else:
        beats_per_bar, beat_value = time_signature.split("/")
        ticks_per_bar = beats_per_bar * (utils.DEFAULT_TICKS_PER_BEAT / (beat_value / 4))
    bars = math.ceil(ticks / ticks_per_bar)
else:
    bars = int(args.length)
tempo = tempo2bpm(tempo)

print("Generating music...")
if args.method == 1:
    generator.generate_music_with_melody_ngrams(
        output_file=output_file,
        bars=bars,
        instrument_melody=args.instrument_melody,
        instrument_harmony=args.instrument_harmony,
        melody_velocity=args.melody_velocity,
        harmony_velocity=args.harmony_velocity,
        with_octave=with_octave,
        only_high_notes_melody=args.only_high_notes_melody,
        only_low_notes_harmony=args.only_low_notes_melody,
        first_note=args.first_note,
        tempo=tempo,
        lengths_flatten_factor=args.flatten_after,
        only_chords=args.only_chords,
        only_arpeggios=args.only_arpeggios,
        more_chords=args.more_chords,
        long_chords=args.long_chords,
    )
elif args.method == 2:
    generator.generate_music_with_harmony_ngrams(
        output_file=output_file,
        bars=bars,
        instrument=args.instrument,
        velocity=args.velocity,
        with_octave=with_octave,
        only_high_notes=args.only_high_notes,
        first_note=args.first_note,
        tempo=tempo,
        lengths_flatten_factor=args.flatten_after,
        start_with_chord=args.start_with_chord,
        strict_time_signature=args.strict_time_signature,
        start_filepath=args.start_filepath,
    )
else:
    generator.generate_music_with_bar_ngrams(
        output_file=output_file,
        bars=bars,
        instrument=args.instrument,
        velocity=args.velocity,
        with_octave=with_octave,
        only_high_notes=args.only_high_notes,
        first_note=args.first_note,
        tempo=tempo,
        start_with_chord=args.start_with_chord,
    )

# ---------------------------- PLAY MUSIC --------------------------------------
print("----------------------")
print(f"File saved as {output_file}.\nPlaying music... Ctrl+C to stop", end="")
new_mid_path = os.path.join(os.getcwd(), output_file)
pygame.mixer.init(44100, -16, 2, 1024)
try:
    clock = pygame.time.Clock()
    pygame.mixer.music.load(new_mid_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(60)
except KeyboardInterrupt:
    pygame.mixer.music.fadeout(1000)
    pygame.mixer.music.stop()
    raise SystemExit
