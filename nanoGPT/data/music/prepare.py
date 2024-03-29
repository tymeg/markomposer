import os
import pickle
import numpy as np

input_file_path = os.path.join(os.path.dirname(__file__), "input_cleanmidi.txt")

with open(input_file_path, "r") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# until_next_note_start note note_length separated by spaces
# in specific format: e.g. I0 N60 L120
input_values = data.split()
print(f"length of dataset in tokens: {len(input_values):,}")
print(f"approximate length of dataset in notes: {len(input_values) // 3:,}")

# vocab consists of values only from input_values
vocab = sorted(list(set(input_values)))
print(vocab)

vocab_size = len(vocab)
print(f"vocab size: {vocab_size:,}")

stoi = {value: i for i, value in enumerate(vocab)}
itos = {i: value for i, value in enumerate(vocab)}


def encode(values):
    return [
        stoi[value] for value in values
    ]  # encoder: take a string, output a list of integers


def decode(tokens):
    return " ".join(
        [itos[token] for token in tokens]
    )  # decoder: take a list of integers, output a string


# create the train and test splits
n = len(input_values)
border = int(n * 0.8)
train_data = input_values[: border]
val_data = input_values[border :]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

with open(os.path.join(os.path.dirname(__file__), "train_ids.txt"), "w") as f:
    f.write(" ".join(map(str, train_ids)))
with open(os.path.join(os.path.dirname(__file__), "val_ids.txt"), "w") as f:
    f.write(" ".join(map(str, val_ids)))

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
