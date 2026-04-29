#include <stdio.h>
#include <cuda.h>

#define ROWS 6
#define COLS 7

// Function to print board
void printBoard(int board[ROWS][COLS]) {
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%d ", board[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Find available row in a column
__device__ int getAvailableRow(int board[ROWS][COLS], int col) {
    for (int i = ROWS - 1; i >= 0; i--) {
        if (board[i][col] == 0)
            return i;
    }
    return -1;
}

// Simple evaluation function
__device__ int evaluateBoard(int board[ROWS][COLS], int player) {
    int score = 0;

    // Simple heuristic: count player pieces
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if (board[i][j] == player)
                score += 1;
        }
    }

    return score;
}

// GPU Kernel: evaluate all columns in parallel
__global__ void evaluateMoves(int *flatBoard, int *scores, int player) {
    int col = threadIdx.x;

    if (col >= COLS) return;

    int board[ROWS][COLS];

    // Copy board from global memory
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            board[i][j] = flatBoard[i * COLS + j];
        }
    }

    int row = getAvailableRow(board, col);

    if (row == -1) {
        scores[col] = -999;
        return;
    }

    board[row][col] = player;

    scores[col] = evaluateBoard(board, player);
}

// Flatten 2D board
void flattenBoard(int board[ROWS][COLS], int *flat) {
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++)
            flat[i * COLS + j] = board[i][j];
}

// Select best move
int selectBestMove(int *scores) {
    int maxScore = -9999;
    int bestMove = 0;

    for (int i = 0; i < COLS; i++) {
        if (scores[i] > maxScore) {
            maxScore = scores[i];
            bestMove = i;
        }
    }

    return bestMove;
}

int main() {
    int board[ROWS][COLS] = {0};
    int flatBoard[ROWS * COLS];
    int scores[COLS];

    int *d_board, *d_scores;

    cudaMalloc(&d_board, sizeof(int) * ROWS * COLS);
    cudaMalloc(&d_scores, sizeof(int) * COLS);

    int currentPlayer = 1;

    for (int turn = 0; turn < 20; turn++) {

        flattenBoard(board, flatBoard);

        cudaMemcpy(d_board, flatBoard, sizeof(int) * ROWS * COLS, cudaMemcpyHostToDevice);

        evaluateMoves<<<1, COLS>>>(d_board, d_scores, currentPlayer);

        cudaMemcpy(scores, d_scores, sizeof(int) * COLS, cudaMemcpyDeviceToHost);

        int move = selectBestMove(scores);

        // Apply move
        for (int i = ROWS - 1; i >= 0; i--) {
            if (board[i][move] == 0) {
                board[i][move] = currentPlayer;
                break;
            }
        }

        printf("Player %d plays column %d\n", currentPlayer, move);
        printBoard(board);

        currentPlayer = (currentPlayer == 1) ? 2 : 1;
    }

    cudaFree(d_board);
    cudaFree(d_scores);

    return 0;
}