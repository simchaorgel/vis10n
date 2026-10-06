# Chess models: encode positions the way training did, pick moves, and play rated matches against Stockfish
# Imports
import math
import threading
from pathlib import Path
from typing import Callable

import chess
import chess.engine
import numpy as np
import torch as t
import torch.nn as nn

from device import device

ROOT = Path(__file__).parent
ENGINES_DIR = ROOT / "engines"

# A game this long is scored as a draw, so two shuffling players can't run forever
MAX_PLIES = 400


# Encoding: must match dataloaders/_lichess_processing.ipynb, which made the training data
#######################################################
# Every (from, to) pair a queen or knight could move between on an empty board: 1792 moves
ALL_MOVES = []
for from_sq in range(64):
    for to_sq in range(64):
        if from_sq == to_sq:
            continue
        dr = abs(to_sq // 8 - from_sq // 8)
        dc = abs(to_sq % 8 - from_sq % 8)
        if dr == 0 or dc == 0 or dr == dc or dr * dc == 2:
            ALL_MOVES.append((from_sq, to_sq))

MOVE_TO_INDEX = {pair: i for i, pair in enumerate(ALL_MOVES)}


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    (12, 8, 8) uint8: planes 0-5 the side to move's pieces, 6-11 the opponent's.
    The board is mirrored when black is to move, so the model always plays 'white'.
    """
    board_tensor = np.zeros((12, 8, 8), dtype=np.uint8)
    if board.turn == chess.BLACK:
        board = board.mirror()
    for square, piece in board.piece_map().items():
        plane = piece.piece_type - 1
        if piece.color == chess.BLACK:
            plane += 6
        board_tensor[plane, chess.square_rank(square), chess.square_file(square)] = 1
    return board_tensor


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Index 0-1791 of a move played from `board`, mirrored like the board when black is to move."""
    from_sq, to_sq = move.from_square, move.to_square
    if board.turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)
    return MOVE_TO_INDEX[(from_sq, to_sq)]


# Playing
#######################################################
def pick_move(model: nn.Module, board: chess.Board) -> tuple[chess.Move, float]:
    """
    The model's favourite legal move, and the probability it gives it out of the legal moves.
    Promotions are always to a queen, since the encoding can't tell promotions apart.
    """
    # Each legal move's index; underpromotions share their index with the queen promotion, so are left out
    legal = {
        move_to_index(move, board): move
        for move in board.legal_moves
        if move.promotion in (None, chess.QUEEN)
    }

    x = t.from_numpy(board_to_tensor(board)).float().unsqueeze(0).to(device)   # (1, 12, 8, 8)
    with t.inference_mode():
        logits = model(x)[0]                                                   # (1792,)

    # Softmax over the legal moves only, so illegal moves get no probability
    indices = list(legal)
    probabilities = t.softmax(logits[indices], dim=0)
    best = probabilities.argmax().item()
    return legal[indices[best]], probabilities[best].item()


def game_status(board: chess.Board) -> str | None:
    """How the game ended, from white's side of the board, or None while it's still going."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return f"draw - {outcome.termination.name.lower().replace('_', ' ')}"
    winner = "white" if outcome.winner == chess.WHITE else "black"
    return f"checkmate - {winner} wins"


# Rating against Stockfish
#######################################################
def find_stockfish() -> Path | None:
    """The first Stockfish executable under engines/."""
    found = sorted(ENGINES_DIR.glob("**/stockfish*.exe"))
    return found[0] if found else None


def elo_estimate(results: list[float], opponent_elo: int) -> dict:
    """
    Rating from a match score, with a 95% range.
    results: the model's points per game (1 win, 0.5 draw, 0 loss)
    Elo is undefined at a score of 0% or 100%, so then only the side it lies on is known.
    """
    n = len(results)
    score = sum(results) / n
    estimate = {"score": score, "elo": None, "low": None, "high": None}
    if score in (0, 1):
        return estimate

    def to_elo(s: float) -> float:
        s = min(max(s, 1e-3), 1 - 1e-3)
        return opponent_elo - 400 * math.log10(1 / s - 1)

    # Standard error of the mean score, from the spread of the actual results
    variance = sum((r - score) ** 2 for r in results) / n
    margin = 1.96 * math.sqrt(variance / n)
    estimate.update(elo=to_elo(score), low=to_elo(score - margin), high=to_elo(score + margin))
    return estimate


def play_match(
    model: nn.Module,
    opponent_elo: int,
    games: int,
    on_update: Callable[[dict], None],
    cancel: threading.Event,
    move_time: float = 0.05,
) -> None:
    """
    Plays `games` games against Stockfish limited to `opponent_elo`, alternating colours.
    on_update gets {"type": "move", ...} after every move and {"type": "game", ...} after every game.
    """
    path = find_stockfish()
    if path is None:
        raise FileNotFoundError("no stockfish in engines/")

    engine = chess.engine.SimpleEngine.popen_uci(str(path))
    try:
        # Stockfish can only be held down to a range of ratings
        lowest, highest = engine.options["UCI_Elo"].min, engine.options["UCI_Elo"].max
        if not lowest <= opponent_elo <= highest:
            raise ValueError(f"elo must be {lowest}-{highest}")
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": opponent_elo, "Threads": 1})
        results = []
        tally = {"wins": 0, "draws": 0, "losses": 0}

        for game in range(games):
            model_color = chess.WHITE if game % 2 == 0 else chess.BLACK
            board = chess.Board()

            while game_status(board) is None and board.ply() < MAX_PLIES:
                if cancel.is_set():
                    return
                if board.turn == model_color:
                    move, _ = pick_move(model, board)
                else:
                    move = engine.play(board, chess.engine.Limit(time=move_time)).move
                board.push(move)
                on_update({
                    "type": "move",
                    "fen": board.fen(),
                    "last": [chess.square_name(move.from_square), chess.square_name(move.to_square)],
                    "model_white": model_color == chess.WHITE,
                })

            # Points from the model's side; a game cut off at MAX_PLIES counts as a draw
            outcome = board.outcome(claim_draw=True)
            if outcome is None or outcome.winner is None:
                points, key = 0.5, "draws"
            elif outcome.winner == model_color:
                points, key = 1.0, "wins"
            else:
                points, key = 0.0, "losses"
            results.append(points)
            tally[key] += 1

            on_update({
                "type": "game",
                "game": game + 1,
                "games": games,
                **tally,
                **elo_estimate(results, opponent_elo),
            })
    finally:
        engine.quit()
