/*
 * Copyright (c) 2005-2015, Brian K. Vogel
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the FreeBSD Project.
 *
 */
#include "Matrix.h"
#include <Accelerate/Accelerate.h>


void resized() {
#ifdef KUMOZU_DEBUG
    std::cout << "Resized." << std::endl;
#endif
}

void _mat_multiply_blas(MatrixF &A, const MatrixF &B, const MatrixF &C, float alpha, float beta)
{
    if (B.extent(1) != C.extent(0))
    {
        error_exit("Error: Inconsistent matrix dimensions! Exiting.");
    }

    const int rows_A = B.extent(0);
    const int cols_A = C.extent(1);
    if ((A.order() != 2) || (A.size() != rows_A * cols_A))
    {
            A.resize(rows_A, cols_A);
    }

    float *backingArrayA = A.get_backing_data();
    const float *backingArrayB = B.get_backing_data();
    const float *backingArrayC = C.get_backing_data();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.extent(0), A.extent(1), B.extent(1), alpha,
                backingArrayB, B.extent(1), backingArrayC, C.extent(1), beta, backingArrayA, A.extent(1));
}





