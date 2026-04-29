# 🎮 GPU vs GPU Connect 4

## 📌 Project Description

This project implements a Connect 4 game where two GPU-based agents compete using parallel computation. Each player evaluates all possible moves simultaneously using CUDA kernels and selects the best move based on a heuristic scoring function.

The goal of this project is to demonstrate how GPUs can be used for decision-making in games by leveraging parallelism instead of traditional sequential CPU logic.

---

## ⚙️ How It Works

* The game board is a 6x7 grid
* Two players (GPU agents) take turns
* Each move is evaluated in parallel using a CUDA kernel
* Each thread corresponds to one possible column
* A heuristic scoring function ranks all moves
* The best move is selected and applied

---

## 💻 Code Description

The core logic is implemented in a CUDA kernel:

* `evaluateMoves`:

  * Each thread evaluates one column
  * Simulates placing a piece
  * Computes a score for that move

* The CPU:

  * Collects scores from GPU
  * Selects the best move
  * Updates the board

This allows all possible moves to be evaluated simultaneously, demonstrating GPU acceleration.

---

## 🧠 Heuristic Used

The scoring function is simple:

* Rewards positions with more player pieces
* Penalizes invalid moves
* Helps select the most favorable board state

---

## 🖥️ Demonstration

The game prints the board state after each move.

Example:
0 0 0 0 0 0 0
0 0 0 1 0 0 0
0 0 0 1 0 0 0
0 0 0 1 0 0 0
0 0 0 2 2 2 0
0 0 2 1 1 0 0
A full sample run is available in:
📄 `demo/sample_output.txt`

---

## 🚀 How to Run

### Requirements

* NVIDIA GPU
* CUDA Toolkit installed

### Compile

```bash
nvcc src/connect4_gpu.cu -o connect4
```

### Run

```bash
./connect4
```

---

## 🚀 Key Highlight

This project demonstrates how GPUs can evaluate multiple game states simultaneously using parallel computation, improving decision-making efficiency.

---

## ⚠️ Challenges Faced

* Managing memory transfer between CPU and GPU
* Handling invalid moves (full columns)
* Debugging GPU kernels

---

## 📌 Future Improvements

* Implement Minimax algorithm on GPU
* Improve heuristic scoring
* Add graphical interface

---

## 👤 Author

Aman Kumar
