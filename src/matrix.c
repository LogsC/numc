#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    // if invalid rows or cols, error -1
    if (rows <= 0 || cols <= 0) {
        return -1;
    }
    // allocate memory for matrix, if space not found, error -1
    matrix *mallocatrix = (matrix *) malloc(sizeof(matrix));
    if (mallocatrix == NULL) {
        return -1;
    }
    int size = rows * cols;
    // allocate memory for data array = # of entries * size of double
    double *data = (double *) malloc(size * sizeof(double));
    // if space not found, error -1
    if (data == NULL) {
        return -1;
    }
    // initialize data array entries to 0.0
    for (int i = 0; i < size; i++) {
        data[i] = 0.0;
    }
    // set mallocatrix values as relevant
    mallocatrix->rows = rows;
    mallocatrix->cols = cols;
    mallocatrix->data = data;
    mallocatrix->ref_cnt = 1;
    mallocatrix->parent = NULL;
    // assign mat ptr to mallocatrix
    *mat = mallocatrix;
    // return 0 success
    return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both are non-positive or if any
 * call to allocate memory in this function fails.
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    // if rows or cols invalid, error -1
    // check for from validity? (cannot assume from != NULL)
    if (rows <= 0 || cols <= 0 || from == NULL) {
        return -1;
    }
    // allocate mat_slice
    matrix *mat_slice = (matrix *) malloc(sizeof(matrix));
    // if space not found, error -1
    if (mat_slice == NULL) {
        return -1;
    }
    // set mat_slice values as relevant
    mat_slice->rows = rows;
    mat_slice->cols = cols;
    // shift from->data by offset
    mat_slice->data = from->data + offset;
    mat_slice->ref_cnt = 1;
    mat_slice->parent = from;
    // increment from ref_cnt
    from->ref_cnt += 1;
    // assign mat ptr to mat_slice
    *mat = mat_slice;
    return 0; // no errors success
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    /* TODO: YOUR CODE HERE */
    // check if mat == NULL (cannot assume mat is not NULL)
    if (mat == NULL) {
        return; // exit function
    }
    if (mat->parent == NULL && mat->ref_cnt <= 1) {
        // if mat is not slice (parent == NULL) and no existing slices (ref_cnt <= 1)
        // free mat->data and mat
        free(mat->data);
        free(mat);
    } else if (mat->ref_cnt > 1) {
        // if mat still has slices (ref_cnt > 1)
        // decrement ref_cnt
        mat->ref_cnt -= 1;
    } else {
        // mat is slice (parent != NULL) and has no existing slices (ref_cnt <= 1)
        // decrement parent->ref_cnt
        mat->parent->ref_cnt -= 1;
        // if parent has no other references including itself (ref_cnt <= 0)
        if (mat->parent->ref_cnt <= 0) {
            // deallocate mat->parent (check parent first for free validity)
            deallocate_matrix(mat->parent);
        }
        // free mat matrix struct
        free(mat);
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    /* TODO: YOUR CODE HERE */
    // get col_size, multiply by row, and add col offset
    int col_size = mat->cols;
    int offset = row * col_size + col;
    return mat->data[offset];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
    // get col_size, multiply by row, and add col offset
    int col_size = mat->cols;
    int offset = row * col_size + col;
    mat->data[offset] = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */
    // multithread / unroll?
    double** data = *(mat->data);
    __m256d vec = _mm256_set1_pd(val);
    int dims = mat->rows * mat-> cols;
    if (dims >= 100000) {
        omp_set_num_threads(8);
    }
    #pragma omp parallel for if (dims >= 100000)
    for (int i = 0; i < dims/16 * 16; i += 16) {
        _mm256_storeu_pd((data + i), vec);
        _mm256_storeu_pd((data + i + 4), vec);
        _mm256_storeu_pd((data + i + 8), vec);
        _mm256_storeu_pd((data + i + 12), vec);
    }
    for (int i = dims/16 * 16; i < dims; i++) {
        data[i] = val;
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    // if dimensions do not match, error 1
    if (result->rows != mat1->rows || result->rows != mat2->rows
            || result->cols != mat1->cols || result->cols != mat2->cols) {
        return 1;
    }
    int size = result->rows * result->cols;
    // multithread / unroll?
    // omp
    if (size >= 100000) {
        omp_set_num_threads(16);
    }
    #pragma omp parallel for if (size >= 100000)
    for (int i = 0; i < size; i++) {
        result->data[i] = mat1->data[i] + mat2->data[i];
    }
    return 0; // success
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    // if dimensions do not match, error 1
    if (result->rows != mat1->rows || result->rows != mat2->rows
            || result->cols != mat1->cols || result->cols != mat2->cols) {
        return 1;
    }
    int size = result->rows * result->cols;
    // multithread / unroll?
    // omp
    if (size >= 100000) {
        omp_set_num_threads(16);
    }
    #pragma omp parallel for if (size >= 100000)
    for (int i = 0; i < size; i++) {
        result->data[i] = mat1->data[i] - mat2->data[i];
    }
    return 0; // success
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    // if dimensions do note match, error 1
    if (result->rows != mat1->rows || result->cols != mat2->cols || mat1->cols != mat2->rows) {
        return 1;
    }
    // create separate case for matrix multiplcation under a certain size?
    int size = result->rows;
    // fill result with 0s
    int res_size = mat->rows * mat->cols;
    result->data = calloc(res_size, sizeof(double);
    if (!result->data) {
        return -1;
    }
    if (size < 32) { // simple naive
        #pragma omp parallel for if (size >= 16)
        for (int i = 0; i < size; i++) {
            for (int k = 0; k < size; k++) {
                for (int j = 0 ; j < size; j++) {
                    result->data[i * size + j] += mat1->data[i * size + k] * mat2->data[k * size + j];
                }
            }
        }
        return 0;
    } else if (size < 200) {
        // allocate the transpose data
        double *tranData  = malloc(size * size * sizeof(double));
        if (tranData == NULL) {
            return -1;
        }
        #pragma omp parallel for
        for (int i = 0; i < size * size; i++) {
            tranData[i] = mat2->data[(i % size) * size + i / size];
        }
        #pragma omp parallel for
        for (int index = 0; index < size * size; index++) {
            double total = 0;
            __m256d sums = _mm256_set1_pd(0);
            int i = index / size * size; int j = index % size * size;
            for (int k = 0 ; k < size / 16 * 16; k = k + 16) {
                sums = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i + k),
                                        _mm256_loadu_pd (tranData + j + k),
                                        sums);
                sums = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i + k + 4),
                                        _mm256_loadu_pd (tranData + j + k + 4),
                                        sums);
                sums = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i + k + 8),
                                        _mm256_loadu_pd (tranData + j + k + 8),
                                        sums);
                sums = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i + k + 12),
                                        _mm256_loadu_pd (tranData + j + k + 12),
                                        sums);
            }
            total = sums[0] + sums[1] + sums[2] + sums[3];
            for (int k = size / 16 * 16; k < size; k++) {
                total += mat1->data[i + k] * tranData[j + k];
            }
            result->data[index] = total;
        }
        free(tranData);
        return 0;
    } else { // size >= 200
        // allocate the transpose data
        double *tranData  = malloc(size * size * sizeof(double));
        if (tranData == NULL) {
            return -1;
        }
        #pragma omp parallel for
        for (int i = 0; i < size * size; i++) {
            tranData[i] = mat2->data[(i % size) * size + i / size];
        }
        #pragma omp parallel for
        for (int index = 0; index < (size * size) / 4 * 4; index = index + 4) {
            // #0
            double total = 0;
            __m256d sums = _mm256_set1_pd(0);
            __m256d secs = _mm256_set1_pd(0);
            int i = index / size * size; int j = index % size * size;
            for (int k = 0 ; k < size / 32 * 32; k = k + 32) {
                sums = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i + k),
                                        _mm256_loadu_pd (tranData + j + k),
                                        sums);
                sums = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i + k + 4),
                                        _mm256_loadu_pd (tranData + j + k + 4),
                                        sums);
                sums = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i + k + 8),
                                        _mm256_loadu_pd (tranData + j + k + 8),
                                        sums);
                sums = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i + k + 12),
                                        _mm256_loadu_pd (tranData + j + k + 12),
                                        sums);
                secs = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i + k + 16),
                                        _mm256_loadu_pd (tranData + j + k + 16),
                                        secs);
                secs = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i + k + 20),
                                        _mm256_loadu_pd (tranData + j + k + 20),
                                        secs);
                secs = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i + k + 24),
                                        _mm256_loadu_pd (tranData + j + k + 24),
                                        secs);
                secs = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i + k + 28),
                                        _mm256_loadu_pd (tranData + j + k + 28),
                                        secs);
            }
            total = sums[0] + sums[1] + sums[2] + sums[3] + secs[0] + secs[1] + secs[2] + secs[3];
            for (int k = size / 32 * 32; k < size; k++) {
                total += mat1->data[i + k] * tranData[j + k];
            }
            result->data[index] = total;
            // #1
            double total1 = 0;
            __m256d sums1 = _mm256_set1_pd(0);
            __m256d secs1 = _mm256_set1_pd(0);
            int i1 = (index + 1) / size * size; int j1 = (index + 1) % size * size;
            for (int k = 0 ; k < size / 32 * 32; k = k + 32) {
                sums1 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i1 + k),
                                         _mm256_loadu_pd (tranData + j1 + k),
                                         sums1);
                sums1 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i1 + k + 4),
                                         _mm256_loadu_pd (tranData + j1 + k + 4),
                                         sums1);
                sums1 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i1 + k + 8),
                                         _mm256_loadu_pd (tranData + j1 + k + 8),
                                         sums1);
                sums1 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i1 + k + 12),
                                         _mm256_loadu_pd (tranData + j1 + k + 12),
                                         sums1);
                secs1 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i1 + k + 16),
                                         _mm256_loadu_pd (tranData + j1 + k + 16),
                                         secs1);
                secs1 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i1 + k + 20),
                                         _mm256_loadu_pd (tranData + j1 + k + 20),
                                         secs1);
                secs1 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i1 + k + 24),
                                         _mm256_loadu_pd (tranData + j1 + k + 24),
                                         secs1);
                secs1 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i1 + k + 28),
                                         _mm256_loadu_pd (tranData + j1 + k + 28),
                                         secs1);
            }
            total1 = sums1[0] + sums1[1] + sums1[2] + sums1[3] + secs1[0] + secs1[1] + secs1[2] + secs1[3];
            for (int k = size / 32 * 32; k < size; k++) {
                total1 += mat1->data[i1 + k] * tranData[j1 + k];
            }
            result->data[index + 1] = total1;
            // #2
            double total2 = 0;
            __m256d sums2 = _mm256_set1_pd(0);
            __m256d secs2 = _mm256_set1_pd(0);
            int i2 = (index + 2) / size * size; int j2 = (index + 2) % size * size;
            for (int k = 0 ; k < size / 32 * 32; k = k + 32) {
                sums2 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i2 + k),
                                         _mm256_loadu_pd (tranData + j2 + k),
                                         sums2);
                sums2 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i2 + k + 4),
                                         _mm256_loadu_pd (tranData + j2 + k + 4),
                                         sums2);
                sums2 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i2 + k + 8),
                                         _mm256_loadu_pd (tranData + j2 + k + 8),
                                         sums2);
                sums2 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i2 + k + 12),
                                         _mm256_loadu_pd (tranData + j2 + k + 12),
                                         sums2);
                secs2 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i2 + k + 16),
                                         _mm256_loadu_pd (tranData + j2 + k + 16),
                                         secs2);
                secs2 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i2 + k + 20),
                                         _mm256_loadu_pd (tranData + j2 + k + 20),
                                         secs2);
                secs2 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i2 + k + 24),
                                         _mm256_loadu_pd (tranData + j2 + k + 24),
                                         secs2);
                secs2 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i2 + k + 28),
                                         _mm256_loadu_pd (tranData + j2 + k + 28),
                                         secs2);
            }
            total2 = sums2[0] + sums2[1] + sums2[2] + sums2[3] + secs2[0] + secs2[1] + secs2[2] + secs2[3];
            for (int k = size / 32 * 32; k < size; k++) {
                total2 += mat1->data[i2 + k] * tranData[j2 + k];
            }
            result->data[index + 2] = total2;
            // #3
            double total3 = 0;
            __m256d sums3 = _mm256_set1_pd(0);
            __m256d secs3 = _mm256_set1_pd(0);
            int i3 = (index + 3) / size * size; int j3 = (index + 3) % size * size;
            for (int k = 0 ; k < size / 32 * 32; k = k + 32){
                sums3 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i3 + k),
                                         _mm256_loadu_pd (tranData + j3 + k),
                                         sums3);
                sums3 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i3 + k + 4),
                                         _mm256_loadu_pd (tranData + j3 + k + 4),
                                         sums3);
                sums3 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i3 + k + 8),
                                         _mm256_loadu_pd (tranData + j3 + k + 8),
                                         sums3);
                sums3 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i3 + k + 12),
                                         _mm256_loadu_pd (tranData + j3 + k + 12),
                                         sums3);
                secs3 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i3 + k + 16),
                                         _mm256_loadu_pd (tranData + j3 + k + 16),
                                         secs3);
                secs3 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i3 + k + 20),
                                         _mm256_loadu_pd (tranData + j3 + k + 20),
                                         secs3);
                secs3 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i3 + k + 24),
                                         _mm256_loadu_pd (tranData + j3 + k + 24),
                                         secs3);
                secs3 = _mm256_fmadd_pd( _mm256_loadu_pd (mat1->data + i3 + k + 28),
                                         _mm256_loadu_pd (tranData + j3 + k + 28),
                                         secs3);
            }
            total3 = sums3[0] + sums3[1] + sums3[2] + sums3[3] + secs3[0] + secs3[1] + secs3[2] + secs3[3];
            for (int k = size / 32 * 32; k < size; k++) {
                total3 += mat1->data[i3 + k] * tranData[j3 + k];
            }
            result->data[index + 3] = total3;
        }
        // tail case
        for (int index = (size * size) / 4 * 4; index < size * size; index++) {
            int i = index / size; int j = index % size;
            double total = 0;
            for (int k = 0; k < size; k++) {
                total += mat1->data[i * size + k] * mat2->data[k * size + j];
            }
            result->data[index] = total;
        }
        free(tranData);
        return 0;
    }
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */
    // if mat is not square or pow is negative, error 1
    if (mat->rows != mat->cols || pow < 0) {
        return 1;
    }
    int size = mat->rows * mat->cols * sizeof(double);
    if (pow == 0) {
        int result_size = result->rows * result->cols;
        if (result_size >= 100000) {
            omp_set_num_threads(16);
        }
        #pragma omp parallel for if (result_size >= 100000)
        for (int i = 0; i < size; ++i) {
            if (i / result->cols == i % result->cols) {
                result->data[i] = 1.0;
            } else {
                result->data[i] = 0.0;
            }
        }
        return 0;
    } else if (pow == 1) {
        memcpy(result->data, mat->data, size);
        return 0;
    }
    matrix *temp1, *temp2;
    if (allocate_matrix(&temp1, result->rows, result->cols) == -1) {
        return -1;
    }
    if (allocate_matrix(&temp2, result->rows, result->cols) == -1) {
        return -1;
    }
    memcpy(temp1->data, mat->data, size);
    int result_size = result->rows * result->cols;
    if (result_size >= 100000) {
        omp_set_num_threads(16);
    }
    #pragma omp parallel for if (result_size >= 100000)
    for (int i = 0; i < size; ++i) {
        if (i / result->cols == i % result->cols) {
            result->data[i] = 1.0;
        } else {
            result->data[i] = 0.0;
        }
    }
    double **swap;
    while (pow > 0) {
        if (pow % 2 == 1) {
            mul_matrix(temp2, result, temp1);
            swap = result->data;
            result->data = temp2->data;
            temp2->data = swap;
        }
        pow = pow / 2;
        if (pow > 0) {
            mul_matrix(temp2, temp1, temp1);
            swap = temp1->data;
            temp1->data = temp2->data;
            temp2->data = swap;
        }
    }
    deallocate_matrix(temp1);
    deallocate_matrix(temp2);
    return 0; // success
    /*
    // calculate size of result
    int size = result->rows * result->cols;
    // handle pow = 0, 1 cases
    if (pow == 0) {
        // fill result as identity matrix (mat^0)
        for (int i = 0; i < size; ++i) {
            if (i / result->cols == i % result->cols) {
                result->data[i] = 1.0;
            } else {
                result->data[i] = 0.0;
            }
        }
        return 0;
    } else if (pow == 1) {
        // copy mat into result
        // copy temp data into result data
        for (int i = 0; i < size; ++i) {
            result->data[i] = mat->data[i];
        }
        return 0;
    }
    // create temp matricies to store values to transfer to result
    matrix *temp;
    matrix *temp2;
    // if allocate fail, return error -1
    if (allocate_matrix(&temp, mat->rows, mat->cols) != 0) {
        return -1;
    }
    if (allocate_matrix(&temp2, mat->rows, mat->cols) != 0) {
        return -1;
    }
    // recurse with mat^2 and pow / 2 until pow == 0 or 1
    mul_matrix(temp, mat, mat); // temp = mat^2
    pow_matrix(temp2, temp, pow / 2); // recurse w/ mat^2 and pow/2, store in temp2
    // handle case of odd pow
    if (pow % 2 == 1) {
        mul_matrix(result, temp2, mat); // multiply by mat 1 more time
    } else {
        // copy mat into result
        for (int i = 0; i < size; i++) {
            result->data[i] = mat->data[i];
        }
    }
    // deallocate temp and temp2
    deallocate_matrix(temp);
    deallocate_matrix(temp2);
    return 0; // success
     */
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    // if result and mat do not match, return 1
    if (result->rows != mat->rows || result->cols != mat->cols) {
        return 1;
    }
    int mHeight = mat->rows;
    int mWidth = mat->cols;
    int rHeight = result->rows;
    int rWidth = result->cols;
    int size = mHeight * mWidth;
    __m256d zeros = _mm256_set1_pd(0);
    if (size >= 100000) {
        omp_set_num_threads(16);
    }
    #pragma omp parallel for if (size >= 100000)
    for (int i = 0; i < size/16 * 16; i += 16) {
        _mm256_storeu_pd((result->data + i + 0),
        _mm256_sub_pd(zeros, _mm256_loadu_pd(mat->data + i + 0)));

        _mm256_storeu_pd((result->data + i + 4),
        _mm256_sub_pd(zeros, _mm256_loadu_pd(mat->data + i + 4)));

        _mm256_storeu_pd((result->data + i + 8),
        _mm256_sub_pd(zeros, _mm256_loadu_pd(mat->data + i + 8)));

        _mm256_storeu_pd((result->data + i + 12),
        _mm256_sub_pd(zeros, _mm256_loadu_pd(mat->data + i + 12)));
    }
    for (int i = size/16 * 16; i < size; i++) {
        result->data[i] = 0 - mat->data[i];
    }
    return 0; // success
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    // if result and mat dimensions do not match, throw error 1
    if (result->rows != mat->rows || result->cols != mat->cols) {
        return 1;
    }
    int mHeight = mat->rows;
    int mWidth = mat->cols;
    int rHeight = result->rows;
    int rWidth = result->cols;
    int size = mHeight * mWidth;
    __m256d zeros = _mm256_set1_pd(0);
    if (size >= 100000) {
        omp_set_num_threads(8);
    }
    #pragma omp parallel for if (size >= 100000)
    for (int i = 0; i < size/16 * 16; i += 16) {
        __m256d m1_v0 = _mm256_loadu_pd(mat->data + i + 0);
        __m256d dif_v0 = _mm256_sub_pd(zeros, m1_v0);
        __m256d abs_v0 = _mm256_max_pd(dif_v0, m1_v0);
        _mm256_storeu_pd((result->data + i + 0), abs_v0);

        __m256d m1_v1 = _mm256_loadu_pd(mat->data + i + 4);
        __m256d dif_v1 = _mm256_sub_pd(zeros, m1_v1);
        __m256d abs_v1 = _mm256_max_pd(dif_v1, m1_v1);
        _mm256_storeu_pd((result->data + i + 4), abs_v1);

        __m256d m1_v2 = _mm256_loadu_pd(mat->data + i + 8);
        __m256d dif_v2 = _mm256_sub_pd(zeros, m1_v2);
        __m256d abs_v2 = _mm256_max_pd(dif_v2, m1_v2);
        _mm256_storeu_pd((result->data + i + 8), abs_v2);

        __m256d m1_v3 = _mm256_loadu_pd(mat->data + i + 12);
        __m256d dif_v3 = _mm256_sub_pd(zeros, m1_v3);
        __m256d abs_v3 = _mm256_max_pd(dif_v3, m1_v3);
        _mm256_storeu_pd((result->data + i + 12), abs_v3);
    }
    for (int i = size/16 * 16; i < size; i++) {
        double val = mat->data[i];
        result->data[i] = (val >= 0) ? val : -val;
    }
    return 0; // success
}