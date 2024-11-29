__global__ void blurKernelTiled(unsigned char *in, unsigned char *out, int w, int h, int TILE_SIZE, int BLUR_SIZE) {
    __shared__ unsigned char sharedTile[32 + 2 * 4][32 + 2 * 4]; // Example for TILE_SIZE = 32, BLUR_SIZE = 4

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    int sharedCol = threadIdx.x + BLUR_SIZE;
    int sharedRow = threadIdx.y + BLUR_SIZE;

    // Load tile into shared memory, including halo regions
    if (row < h && col < w) {
        sharedTile[sharedRow][sharedCol] = in[row * w + col];

        // Handle boundary regions
        if (threadIdx.x < BLUR_SIZE) {
            sharedTile[sharedRow][threadIdx.x] = (col >= BLUR_SIZE) ? in[row * w + col - BLUR_SIZE] : 0;
            sharedTile[sharedRow][sharedCol + TILE_SIZE] = (col + TILE_SIZE < w) ? in[row * w + col + TILE_SIZE] : 0;
        }
        if (threadIdx.y < BLUR_SIZE) {
            sharedTile[threadIdx.y][sharedCol] = (row >= BLUR_SIZE) ? in[(row - BLUR_SIZE) * w + col] : 0;
            sharedTile[sharedRow + TILE_SIZE][sharedCol] = (row + TILE_SIZE < h) ? in[(row + TILE_SIZE) * w + col] : 0;
        }
    }
    __syncthreads();

    // Perform computation
    if (row < h && col < w) {
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                pixVal += sharedTile[sharedRow + blurRow][sharedCol + blurCol];
                ++pixels;
            }
        }
        out[row * w + col] = (unsigned char)(pixVal / pixels);
    }
}
