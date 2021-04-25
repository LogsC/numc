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
    int size = mat->rows * mat->cols;
    omp_set_num_threads(16);
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        mat->data[i] = val;
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
    double *r_data = result->data;
    double *m1_data = mat1->data;
    double *m2_data = mat2->data;
    // multithread / unroll?
    // omp
    omp_set_num_threads(16);
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        r_data[i] = m1_data[i] + m2_data[i];
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
    double *r_data = result->data;
    double *m1_data = mat1->data;
    double *m2_data = mat2->data;
    // multithread / unroll?
    // omp
    omp_set_num_threads(16);
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        r_data[i] = m1_data[i] - m2_data[i];
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
    // OPT?
    // fill result with 0s
    fill_matrix(result, 0.0);
    // multithread / unroll?
    for (int i = 0; i < result->rows; ++i) {
        for (int j = 0; j < result->cols; ++j) {
            // parse result matrix, use omp to obtain row * col sum
            double curr_sum = 0.0;
            #pragma omp parallel reduction(+:curr_sum)
            {
                #pragma omp for
                for (int k = 0; k < mat1->cols; k++) {
                    // curr sum += mat1[row][k] * mat2[k][col]
                    // i = result row = mat1 row
                    // j = result col = mat2 col
                    // k = mat1 col = mat2 row
                    curr_sum += get(mat1, i, k) * get(mat2, k, j);
                }
            }
            set(result, i, j, curr_sum);
        }
    }
    return 0; // success
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
    int size = result->rows * result->cols;
    for (int i = 0; i < size; i++) {
        result->data[i] = mat->data[i] * -1.0;
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
    int size = result->rows * result->cols;
    for (int i = 0; i < size; i++) {
        if (mat->data[i] >= 0) {
            result->data[i] = mat->data[i];
        } else {
            result->data[i] = mat->data[i] * -1.0;
        }
    }
    return 0; // success
}

