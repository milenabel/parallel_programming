#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    double clkbegin, clkend;
    double t, tmax;
    int i, it, m;
    int myid, nprocs;
    int MsgLen, MaxMsgLen, Niter;
    MPI_Status status[2];
    MPI_Request request[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Define the message sizes in doubles
    int msgLen[] = {1, 8, 64, 512, 4096, 32768, 262144, 1048576};
    int msgCount = sizeof(msgLen) / sizeof(msgLen[0]);

    // Allocate buffers for the largest message size
    MaxMsgLen = 1048576;
    double *in = (double *)malloc(MaxMsgLen * sizeof(double));
    double *out = (double *)malloc(MaxMsgLen * sizeof(double));
    for (i = 0; i < MaxMsgLen; i++) out[i] = i;

    for (m = 0; m < msgCount; m++) {
        MsgLen = msgLen[m];
        Niter = 1000000 / (100 + MsgLen / 256);
        if (Niter == 0) Niter = 1; // Ensure Niter is never zero
        MPI_Barrier(MPI_COMM_WORLD);

        clkbegin = MPI_Wtime();
        for (it = 0; it < nprocs * Niter; it++) {
            // Non-blocking send/receive
            MPI_Isend(out, MsgLen, MPI_DOUBLE, (myid + 1) % nprocs, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Irecv(in, MsgLen, MPI_DOUBLE, (myid + nprocs - 1) % nprocs, 0, MPI_COMM_WORLD, &request[1]);

            // Wait for both operations to complete
            MPI_Waitall(2, request, status);

            MPI_Isend(in, MsgLen, MPI_DOUBLE, (myid + 1) % nprocs, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Irecv(out, MsgLen, MPI_DOUBLE, (myid + nprocs - 1) % nprocs, 0, MPI_COMM_WORLD, &request[1]);

            // Wait for both operations to complete
            MPI_Waitall(2, request, status);
        }
        clkend = MPI_Wtime();
        t = clkend - clkbegin;
        MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (myid == 0) {
            if (tmax < 1e9) {
                printf("Ring Communication: Message Size = %d; Time = %.3f nanoseconds\n",
                       MsgLen, (tmax * 1e9) / (nprocs * Niter));
            } else {
                printf("Ring Communication: Message Size = %d; Timing exceeded threshold.\n", MsgLen);
            }
        }
    }

    free(in);
    free(out);
    MPI_Finalize();
    return 0;
}
