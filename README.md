# 🎮 GPU vs GPU Connect 4

This project demonstrates GPU-based parallel decision-making in games.

---

## 📌 Project Description

This project implements a Connect 4 game in which two GPU-based agents compete against each other. The goal is to demonstrate how parallel computing on GPUs can be used for decision-making in games.

In this implementation, each player evaluates all possible moves simultaneously using a CUDA kernel. Each thread is responsible for evaluating one possible column in the game board. The move that results in the highest score is selected and executed.

Unlike traditional CPU-based approaches that evaluate moves sequentially, this project leverages GPU parallelism to evaluate multiple game states at the same time.

The system alternates turns between two GPU agents until a winning condition (four connected pieces) is reached or the board is full.

---

## ⚙️ How It Works

* The game board is a 6x7 grid
* Two GPU agents act as players
* Each move is evaluated in parallel using CUDA
* Each GPU thread evaluates one column
* A heuristic scoring function ranks all possible moves
* The best move is selected and applied
* Players alternate turns until the game ends

---

## 💻 Code Description

The project consists of several key components:

### 1. Board Representation

The board is implemented as a 6x7 2D array:

* `0` = empty cell
* `1` = Player 1 (GPU 1)
* `2` = Player 2 (GPU 2)

---

### 2. GPU Kernel: `evaluateMoves`

This is the core part of the program.

* Each thread corresponds to one column
* The kernel performs:

  1. Copies the board into local memory
  2. Simulates placing a piece in that column
  3. Evaluates the board using a heuristic function
  4. Stores the score for that move

This allows all possible moves to be evaluated simultaneously using GPU parallelism.

---

### 3. Heuristic Function

The function `evaluateBoard` assigns scores based on:

* Number of player pieces
* Validity of the move

This helps determine the most favorable move.

---

### 4. Move Selection (CPU Side)

After GPU execution:

* Scores are copied back to CPU memory
* The CPU selects the move with the highest score
* The board is updated accordingly

---

### 5. Turn Management

The game alternates between Player 1 and Player 2 until:

* A player wins (connects 4 pieces)
* Or the board reaches maximum turns

---

## 🖥️ Demonstration and Visualization

The game is displayed using a text-based console output.

### Board Representation Example

```
0 0 0 0 0 0 0
0 0 0 1 0 0 0
0 0 0 1 0 0 0
0 0 0 1 0 0 0
0 0 0 2 2 2 0
0 0 2 1 1 0 0
```

* `1` represents Player 1 (GPU 1)
* `2` represents Player 2 (GPU 2)

---

### Game Flow

* Each turn displays:

  * Which player is making a move
  * The selected column
  * The updated board state

---

### Visualization Design

Although the implementation uses console output, it is structured to clearly show:

* Step-by-step game progression
* Player decisions
* Winning patterns (vertical, horizontal, diagonal)

---

### Sample Execution

A full example game run is available in:

📄 `demo/sample_output.txt`

This file demonstrates how the game evolves and how GPU-based decision-making affects gameplay.

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

## 🚀 Key Insight

This project demonstrates that GPUs can evaluate multiple possible game states simultaneously, making them highly effective for AI decision-making compared to traditional sequential CPU approaches.

---

## ⚠️ Challenges Faced

* Managing memory transfer between CPU and GPU
* Handling invalid moves (full columns)
* Debugging CUDA kernels

---

## 📌 Future Improvements

* Implement Minimax algorithm on GPU
* Improve heuristic scoring
* Add graphical interface (GUI)

---

## 👤 Author

Aman Kumar
