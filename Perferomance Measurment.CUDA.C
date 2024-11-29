void measureKernelExecutionTime(dim3 gridSize, dim3 blockSize, unsigned char *d_in, unsigned char *d_out, int w, int h, int TILE_SIZE, int BLUR_SIZE, bool tiled) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (tiled) {
        blurKernelTiled<<<gridSize, blockSize>>>(d_in, d_out, w, h, TILE_SIZE, BLUR_SIZE);
    } else {
        blurKernel<<<gridSize, blockSize>>>(d_in, d_out, w, h, BLUR_SIZE);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time (%s): %f ms\n", tiled ? "Tiled" : "Non-Tiled", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
