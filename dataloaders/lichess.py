# Load the processed lichess positions (made by _lichess_processing.ipynb)
# Imports
from pathlib import Path

import numpy as np
import torch as t
from torch.utils.data import TensorDataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROCESSED_DIR = DATA_DIR / "chess" / "processed"


def get_lichess(trainset_size: int = 900_000, testset_size: int = 100_000) -> tuple[TensorDataset, TensorDataset]:
    """
    Returns (trainset, testset) of (board, move) pairs.
    board: (8, 8) uint8 piece numbers - 0 empty, 1-6 the side to move's pieces, 7-12 the opponent's.
           Expand each batch to the model's (12, 8, 8) planes on the GPU (one_hot, drop the empty class)
    move: int64 index 0-1791
    The split is by game, so no game has positions in both sets.
    """
    boards = np.load(PROCESSED_DIR / "boards.npy")
    moves = np.load(PROCESSED_DIR / "moves.npy")
    game_ids = np.load(PROCESSED_DIR / "game_ids.npy")

    # Test set is the last positions in the file, moved back to the start of a game
    # so no game is split across train and test (game_ids are in ascending order)
    cut = max(len(boards) - testset_size, 0)
    cut = np.searchsorted(game_ids, game_ids[cut])

    # Train set is taken from the start of the file, never past the cut
    train_end = min(trainset_size, cut)

    boards = t.from_numpy(boards)
    moves = t.from_numpy(moves).long()  # cross_entropy needs int64 targets

    trainset = TensorDataset(boards[:train_end], moves[:train_end])
    testset = TensorDataset(boards[cut:], moves[cut:])

    return trainset, testset
