import os
import sys
import pickle
import numpy as np

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

sys.path.append(project_dir)
from gen_music import generator
from utils import ALL_NOTES_COUNT

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

with open(input_file_path, "r") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# note,note_length,until_next_note_start separated by spaces
input_notes = data.split()
print(f"length of dataset in notes: {len(input_notes):,}")

all_notes = [
    ",".join([str(note), str(note_length), str(until_next_note_start)])
    for note in range(ALL_NOTES_COUNT)
    for note_length in generator.note_lengths_range
    for until_next_note_start in generator.until_next_range
]

vocab_size = (
    ALL_NOTES_COUNT
    * len(generator.note_lengths_range)
    * len(generator.until_next_range)
)
print(f"vocab size: {vocab_size:,}")

stoi = {note: i for i, note in enumerate(all_notes)}
itos = {i: note for i, note in enumerate(all_notes)}


def encode(notes):
    return [
        stoi[note] for note in notes
    ]  # encoder: take a string, output a list of integers


def decode(tokens):
    return ",".join(
        [itos[token] for token in tokens]
    )  # decoder: take a list of integers, output a string


# create the train and test splits
n = len(input_notes)
train_data = input_notes[: int(n * 0.9)]
val_data = input_notes[int(n * 0.9) :]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint32)
val_ids = np.array(val_ids, dtype=np.uint32)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# save the meta information as well, to help us encode/decode later
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
