#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "slenet_params.h"

#define IMAGES_PATH "data/t10k-images.idx3-ubyte"
#define LABELS_PATH "data/t10k-labels.idx1-ubyte"

#define INSIZE 28

typedef struct mnist_data {

    double data[INSIZE][INSIZE];
    unsigned int label;

} mnist_data;

static unsigned int mnist_bin_to_int(char *tmp) {

    unsigned int num;
    memcpy(&num, tmp, sizeof(num));

    unsigned int result = ((num >> 24) & 0x000000ff) |
                 ((num >> 8) & 0x0000ff00) |
                 ((num << 8) & 0x00ff0000) |
                 ((num << 24) & 0xff000000);

    return result;
}

/**** CONVOLUTIONAL LAYER ****/

__global__ void kernel_conv( double input[28][28], float output[6][24][24], float weight[6][5][5], float bias[6] ){
    size_t channel = blockIdx. x;
    size_t tx = threadIdx. x;
    size_t ty = threadIdx. y;
  
    __shared__ float sweight[ 5 ][ 5 ];
    __shared__ float sinput[ 28 ][ 28 ];
    __shared__ float sbias[ 6 ];
  
    if ( tx < 5 && ty < 5 ){
      sweight[ tx ][ ty ] = weight[ channel ][ tx ][ ty ];
      if( tx == 0 && ty == 0 ){
        sbias[ channel ] = bias[ channel ];
      }
    }
    sinput[ tx ][ ty ] = input[ tx ][ ty ];
    __syncthreads();
  
    float sum = 0.0;
    if ( tx < 24 && ty < 24 ){
        sum += sinput[ tx ][ ty ] * sweight[ 0 ][ 0 ];
        sum += sinput[ tx ][ty+1] * sweight[ 0 ][ 1 ];
        sum += sinput[ tx ][ty+2] * sweight[ 0 ][ 2 ];
        sum += sinput[ tx ][ty+3] * sweight[ 0 ][ 3 ];
        sum += sinput[ tx ][ty+4] * sweight[ 0 ][ 4 ];
        sum += sinput[tx+1][ ty ] * sweight[ 1 ][ 0 ];
        sum += sinput[tx+1][ty+1] * sweight[ 1 ][ 1 ];
        sum += sinput[tx+1][ty+2] * sweight[ 1 ][ 2 ];
        sum += sinput[tx+1][ty+3] * sweight[ 1 ][ 3 ];
        sum += sinput[tx+1][ty+4] * sweight[ 1 ][ 4 ];
        sum += sinput[tx+2][ ty ] * sweight[ 2 ][ 0 ];
        sum += sinput[tx+2][ty+1] * sweight[ 2 ][ 1 ];
        sum += sinput[tx+2][ty+2] * sweight[ 2 ][ 2 ];
        sum += sinput[tx+2][ty+3] * sweight[ 2 ][ 3 ];
        sum += sinput[tx+2][ty+4] * sweight[ 2 ][ 4 ];
        sum += sinput[tx+3][ ty ] * sweight[ 3 ][ 0 ];
        sum += sinput[tx+3][ty+1] * sweight[ 3 ][ 1 ];
        sum += sinput[tx+3][ty+2] * sweight[ 3 ][ 2 ];
        sum += sinput[tx+3][ty+3] * sweight[ 3 ][ 3 ];
        sum += sinput[tx+3][ty+4] * sweight[ 3 ][ 4 ];
        sum += sinput[tx+4][ ty ] * sweight[ 4 ][ 0 ];
        sum += sinput[tx+4][ty+1] * sweight[ 4 ][ 1 ];
        sum += sinput[tx+4][ty+2] * sweight[ 4 ][ 2 ];
        sum += sinput[tx+4][ty+3] * sweight[ 4 ][ 3 ];
        sum += sinput[tx+4][ty+4] * sweight[ 4 ][ 4 ];
        sum += sbias[ channel ];
        output[ channel ][ tx ][ ty ] = ( 1.0 / ( 1.0 + exp( -sum ) ) );
    }

    return;
}

/**** SUBSAMPLING LAYER ****/

__global__ void kernel_ss(float input[6][24][24], float output[6][6][6], float weight[1][4][4], float bias[1] ){
    size_t channel = threadIdx. z;
    size_t tx = threadIdx. x;
    size_t ty = threadIdx. y;
  
    __shared__ float sweight[ 4 ][ 4 ];
    __shared__ float sinput[ 6 ][ 24 ][ 24 ];
    __shared__ float sbias;
  
    sinput[ channel ][ tx * 4 + 0 ][ ty * 4 + 0 ] = input[ channel ][ tx * 4 + 0 ][ ty * 4 + 0 ];
    sinput[ channel ][ tx * 4 + 0 ][ ty * 4 + 1 ] = input[ channel ][ tx * 4 + 0 ][ ty * 4 + 1 ];
    sinput[ channel ][ tx * 4 + 0 ][ ty * 4 + 2 ] = input[ channel ][ tx * 4 + 0 ][ ty * 4 + 2 ];
    sinput[ channel ][ tx * 4 + 0 ][ ty * 4 + 3 ] = input[ channel ][ tx * 4 + 0 ][ ty * 4 + 3 ];
    sinput[ channel ][ tx * 4 + 1 ][ ty * 4 + 0 ] = input[ channel ][ tx * 4 + 1 ][ ty * 4 + 0 ];
    sinput[ channel ][ tx * 4 + 1 ][ ty * 4 + 1 ] = input[ channel ][ tx * 4 + 1 ][ ty * 4 + 1 ];
    sinput[ channel ][ tx * 4 + 1 ][ ty * 4 + 2 ] = input[ channel ][ tx * 4 + 1 ][ ty * 4 + 2 ];
    sinput[ channel ][ tx * 4 + 1 ][ ty * 4 + 3 ] = input[ channel ][ tx * 4 + 1 ][ ty * 4 + 3 ];
    sinput[ channel ][ tx * 4 + 2 ][ ty * 4 + 0 ] = input[ channel ][ tx * 4 + 2 ][ ty * 4 + 0 ];
    sinput[ channel ][ tx * 4 + 2 ][ ty * 4 + 1 ] = input[ channel ][ tx * 4 + 2 ][ ty * 4 + 1 ];
    sinput[ channel ][ tx * 4 + 2 ][ ty * 4 + 2 ] = input[ channel ][ tx * 4 + 2 ][ ty * 4 + 2 ];
    sinput[ channel ][ tx * 4 + 2 ][ ty * 4 + 3 ] = input[ channel ][ tx * 4 + 2 ][ ty * 4 + 3 ];
    sinput[ channel ][ tx * 4 + 3 ][ ty * 4 + 0 ] = input[ channel ][ tx * 4 + 3 ][ ty * 4 + 0 ];
    sinput[ channel ][ tx * 4 + 3 ][ ty * 4 + 1 ] = input[ channel ][ tx * 4 + 3 ][ ty * 4 + 1 ];
    sinput[ channel ][ tx * 4 + 3 ][ ty * 4 + 2 ] = input[ channel ][ tx * 4 + 3 ][ ty * 4 + 2 ];
    sinput[ channel ][ tx * 4 + 3 ][ ty * 4 + 3 ] = input[ channel ][ tx * 4 + 3 ][ ty * 4 + 3 ];
  
    if ( tx < 4 && ty < 4 ){
      sweight[ tx ][ ty ] = weight[ 0 ][ tx ][ ty ];
      if ( tx == 0 && ty == 0 ) sbias = bias[ 0 ];
    }
    __syncthreads();
    float sum = 0.0;
    sum += sweight[ 0 ][ 0 ] * sinput[ channel ][ tx * 4 + 0 ][ ty * 4 + 0 ];
    sum += sweight[ 0 ][ 1 ] * sinput[ channel ][ tx * 4 + 0 ][ ty * 4 + 1 ];
    sum += sweight[ 0 ][ 2 ] * sinput[ channel ][ tx * 4 + 0 ][ ty * 4 + 2 ];
    sum += sweight[ 0 ][ 3 ] * sinput[ channel ][ tx * 4 + 0 ][ ty * 4 + 3 ];
    sum += sweight[ 1 ][ 0 ] * sinput[ channel ][ tx * 4 + 1 ][ ty * 4 + 0 ];
    sum += sweight[ 1 ][ 1 ] * sinput[ channel ][ tx * 4 + 1 ][ ty * 4 + 1 ];
    sum += sweight[ 1 ][ 2 ] * sinput[ channel ][ tx * 4 + 1 ][ ty * 4 + 2 ];
    sum += sweight[ 1 ][ 3 ] * sinput[ channel ][ tx * 4 + 1 ][ ty * 4 + 3 ];
    sum += sweight[ 2 ][ 0 ] * sinput[ channel ][ tx * 4 + 2 ][ ty * 4 + 0 ];
    sum += sweight[ 2 ][ 1 ] * sinput[ channel ][ tx * 4 + 2 ][ ty * 4 + 1 ];
    sum += sweight[ 2 ][ 2 ] * sinput[ channel ][ tx * 4 + 2 ][ ty * 4 + 2 ];
    sum += sweight[ 2 ][ 3 ] * sinput[ channel ][ tx * 4 + 2 ][ ty * 4 + 3 ];
    sum += sweight[ 3 ][ 0 ] * sinput[ channel ][ tx * 4 + 3 ][ ty * 4 + 0 ];
    sum += sweight[ 3 ][ 1 ] * sinput[ channel ][ tx * 4 + 3 ][ ty * 4 + 1 ];
    sum += sweight[ 3 ][ 2 ] * sinput[ channel ][ tx * 4 + 3 ][ ty * 4 + 2 ];
    sum += sweight[ 3 ][ 3 ] * sinput[ channel ][ tx * 4 + 3 ][ ty * 4 + 3 ];
    sum += sbias;
    output[ channel ][ tx ][ ty ] = ( 1.0 / ( 1.0 + exp( -sum ) ) );
    return;
}

__global__ void kernel_ss_bias(float pre_output[6][6][6], float bias[1]){
    size_t channel = threadIdx. z;
    size_t x       = threadIdx. x;
    size_t y       = threadIdx. y;
    pre_output[ channel ][ x ][ y ] += bias[ 0 ];
    return;
}
__global__ void kernel_ss_sigmoid(float pre_output[6][6][6], float output[6][6][6]){
    size_t channel = threadIdx. z;
    size_t x       = threadIdx. x;
    size_t y       = threadIdx. y;
    output[ channel ][ x ][ y ] = ( 1.0 / ( 1.0 + exp( -pre_output[ channel ][ x ][ y ] ) ) );
    return;
}

/**** FULLY-CONNECTED LAYER ****/

__global__ void kernel_fc(float input[6][6][6], float pre_output[10], float weight[10][6][6][6]){
    size_t channel = blockIdx. x;
  
    size_t tx = threadIdx. x;
    size_t ty = threadIdx. y;
    size_t tz = threadIdx. z;
  
    __shared__ float temp[6][6][6];
    __shared__ float temp2[6];
    temp[ tx ][ ty ][ tz ] = input[ tx ][ ty ][ tz ] * weight[ channel ][ tx ][ ty ][ tz ];
    __syncthreads();
    temp2[ tx ] = 0.0;
    if ( ty == 0 && tz == 0 ){
        temp2[ tx ] += temp[ tx ][ 0 ][ 0 ];
        temp2[ tx ] += temp[ tx ][ 0 ][ 1 ];
        temp2[ tx ] += temp[ tx ][ 0 ][ 2 ];
        temp2[ tx ] += temp[ tx ][ 0 ][ 3 ];
        temp2[ tx ] += temp[ tx ][ 0 ][ 4 ];
        temp2[ tx ] += temp[ tx ][ 0 ][ 5 ];
        temp2[ tx ] += temp[ tx ][ 1 ][ 0 ];
        temp2[ tx ] += temp[ tx ][ 1 ][ 1 ];
        temp2[ tx ] += temp[ tx ][ 1 ][ 2 ];
        temp2[ tx ] += temp[ tx ][ 1 ][ 3 ];
        temp2[ tx ] += temp[ tx ][ 1 ][ 4 ];
        temp2[ tx ] += temp[ tx ][ 1 ][ 5 ];
        temp2[ tx ] += temp[ tx ][ 2 ][ 0 ];
        temp2[ tx ] += temp[ tx ][ 2 ][ 1 ];
        temp2[ tx ] += temp[ tx ][ 2 ][ 2 ];
        temp2[ tx ] += temp[ tx ][ 2 ][ 3 ];
        temp2[ tx ] += temp[ tx ][ 2 ][ 4 ];
        temp2[ tx ] += temp[ tx ][ 2 ][ 5 ];
        temp2[ tx ] += temp[ tx ][ 3 ][ 0 ];
        temp2[ tx ] += temp[ tx ][ 3 ][ 1 ];
        temp2[ tx ] += temp[ tx ][ 3 ][ 2 ];
        temp2[ tx ] += temp[ tx ][ 3 ][ 3 ];
        temp2[ tx ] += temp[ tx ][ 3 ][ 4 ];
        temp2[ tx ] += temp[ tx ][ 3 ][ 5 ];
        temp2[ tx ] += temp[ tx ][ 4 ][ 0 ];
        temp2[ tx ] += temp[ tx ][ 4 ][ 1 ];
        temp2[ tx ] += temp[ tx ][ 4 ][ 2 ];
        temp2[ tx ] += temp[ tx ][ 4 ][ 3 ];
        temp2[ tx ] += temp[ tx ][ 4 ][ 4 ];
        temp2[ tx ] += temp[ tx ][ 4 ][ 5 ];
        temp2[ tx ] += temp[ tx ][ 5 ][ 0 ];
        temp2[ tx ] += temp[ tx ][ 5 ][ 1 ];
        temp2[ tx ] += temp[ tx ][ 5 ][ 2 ];
        temp2[ tx ] += temp[ tx ][ 5 ][ 3 ];
        temp2[ tx ] += temp[ tx ][ 5 ][ 4 ];
        temp2[ tx ] += temp[ tx ][ 5 ][ 5 ];
    }
    __syncthreads();
    if ( tx == 0 && ty == 0 && tz == 0 ){
        float sum = 0.0;
        sum += temp2[ 0 ];
        sum += temp2[ 1 ];
        sum += temp2[ 2 ];
        sum += temp2[ 3 ];
        sum += temp2[ 4 ];
        sum += temp2[ 5 ];
        pre_output[ channel ] = sum;
    }
    return;
}
  
__global__ void kernel_fc_bias(float pre_output[10], float bias[10]){
    size_t x = threadIdx. x;
    pre_output[ x ] += bias[ x ];
    return;
}
  
__global__ void kernel_fc_sigmoid(float pre_output[10], float output[10]){
    size_t x = threadIdx. x;
    output[ x ] = ( 1.0 / ( 1.0 + exp( -pre_output[ x ] ) ) );
    return;
}

/**** MNIST LOAD ****/

static int mnist_load(const char *image_filename, const char *label_filename, mnist_data **data_set, unsigned int *count) {
    int error = 1;
    FILE *images = fopen(image_filename, "rb");
    FILE *labels = fopen(label_filename, "rb");

    unsigned int res;
    char val[4];
    *count = 0;

    //MAGIC NUMBERS
    res = fread(&val, 4, 1, images);
    int iMN = mnist_bin_to_int(val);

    res = fread(&val, 4, 1, labels);
    int lMN = mnist_bin_to_int(val);
    
    //NUMBER OF ITEMS
    res = fread(&val, 4, 1, images);
    int imgNum = mnist_bin_to_int(val);

    res = fread(&val, 4, 1, labels);
    int itemNum = mnist_bin_to_int(val);

    //NUM OF ROWS AND COLUMNS
    res = fread(&val, 4, 1, images);
    int rowNum = mnist_bin_to_int(val);
    
    res = fread(&val, 4, 1, images);
    int colNum = mnist_bin_to_int(val);

    //LOAD DATA SET
    int pixel;
    unsigned char label;

    *data_set = (mnist_data*)malloc(sizeof(mnist_data) * 10000);
    
    int num_images = 0;
    int a = 0;
    while (a < 10000) {
        for (int i=0; i < INSIZE; i++) {
            for (int j=0; j < INSIZE; j++) {
                res = fread(&pixel, 1, 1, images);
                (*data_set)[a].data[i][j] = (double) pixel/255.0;
            }
        }
        num_images++;
        label = fgetc(labels);
        (*data_set)[a].label = (int) label;
        a++;
    }

    //CLOSE FILES
    fclose(images);
    fclose(labels);

    *count = num_images;
    error = 0;
    return error;
}

class Layer {

public:

    int M, N, O;
    double *input;
    float *pre_output, *output;
    float *pre_output_ss, *output_ss;
    float *pre_output_fc, *output_fc;
    float *weight, *bias;
    float *weight_ss, *bias_ss;
    float *weight_fc, *bias_fc;

    Layer(int M, int N, int O): M{ M }, N{ N }, O{ O }{
        cudaMalloc( &input,         sizeof( double ) * 28 * 28        );
        cudaMalloc( &pre_output,    sizeof( float ) * 6  * 24 * 24   );
        cudaMalloc( &output,        sizeof( float ) * 6  * 24 * 24   );
        cudaMalloc( &pre_output_ss, sizeof( float ) * 6  * 6  * 6    );
        cudaMalloc( &output_ss,     sizeof( float ) * 6  * 6  * 6    );
        cudaMalloc( &pre_output_fc, sizeof( float ) * 10             );
        cudaMalloc( &output_fc,     sizeof( float ) * 10             );
        cudaMalloc( &weight,        sizeof( float ) * 6  * 5  * 5    );
        cudaMalloc( &bias,          sizeof( float ) * 6              );
        cudaMalloc( &weight_ss,     sizeof( float ) * 1  * 4  * 4    );
        cudaMalloc( &bias_ss,       sizeof( float ) * 1              );
        cudaMalloc( &weight_fc,     sizeof( float ) * 10 * 6 * 6 * 6 );
        cudaMalloc( &bias_fc,       sizeof( float ) * 10             );
    }

    ~Layer(){
        cudaFree( input         );
        cudaFree( pre_output    );
        cudaFree( output        );
        cudaFree( pre_output_ss );
        cudaFree( output_ss     );
        cudaFree( pre_output_fc );
        cudaFree( output_fc     );
        cudaFree( weight        );
        cudaFree( bias          );
        cudaFree( weight_ss     );
        cudaFree( bias_ss       );
        cudaFree( weight_fc     );
        cudaFree( bias_fc       );
    }

    int forward_pass( double data[28][28] );

};

int decipher_output( float output[10] ){
    int i = 0, _res;
    float mx = -99.0;
    for( i = 0; i < 10; ++ i ){
      if( mx < output[ i ] ){
        mx = output[ i ];
        _res = i;
      }
    }
  
    return _res;
  }

int Layer::forward_pass( double data[28][28] ){
    cudaMemcpy( input, data, sizeof( double ) * 28 * 28, cudaMemcpyHostToDevice );
  
    dim3 threadsNum( 5, 5, 6 );
  
    kernel_conv<<< 6, dim3( 28, 28 ) >>>( (double (*)[28]) input, (float (*)[24][24]) output, (float (*)[5][5]) weight, bias );
    kernel_ss<<< 1, dim3( 6, 6, 6 ) >>>( (float (*)[24][24]) output, (float (*)[6][6]) output_ss, (float (*)[4][4]) weight_ss, bias_ss );
  
    float temp_pre_output_fc[10];
  
    threadsNum = dim3( 6, 6, 6 );
  
    kernel_fc<<< 10, threadsNum >>>( (float (*)[6][6])output_ss, pre_output_fc, (float (*)[6][6][6]) weight_fc );
    kernel_fc_bias<<< 1, 10 >>>( pre_output_fc, bias_fc );
    kernel_fc_sigmoid<<< 1, 10 >>>( pre_output_fc, output_fc );
  
    cudaMemcpy( temp_pre_output_fc, output_fc, sizeof( float ) * 10, cudaMemcpyDeviceToHost );
    return decipher_output( temp_pre_output_fc );
}

int main(void) {
    int ret;
    unsigned int num;
    mnist_data *data;

    Layer layer(0, 0, 0);

    if (ret = mnist_load(IMAGES_PATH, LABELS_PATH, &data, &num) != 0)
        printf("An error occurred: %d \n", ret);
    else
        printf("test_cnt = %d \n", num);
    
    cudaMemcpy( layer. bias,      c1_bias,   sizeof( float ) * 6,                cudaMemcpyHostToDevice );
    cudaMemcpy( layer. bias_ss,   s2_bias,   sizeof( float ) * 1,                cudaMemcpyHostToDevice );
    cudaMemcpy( layer. bias_fc,   f3_bias,   sizeof( float ) * 10,               cudaMemcpyHostToDevice );
    cudaMemcpy( layer. weight,    c1_weight, sizeof( float ) * 6  * 5  * 5,      cudaMemcpyHostToDevice );
    cudaMemcpy( layer. weight_ss, s2_weight, sizeof( float ) * 1  * 4  * 4,      cudaMemcpyHostToDevice );
    cudaMemcpy( layer. weight_fc, f3_weight, sizeof( float ) * 10 * 6  * 6  * 6, cudaMemcpyHostToDevice );
    
    int i = 0, err = 0;
    for( i = 0; i < num; ++ i ){
        int first_guess = layer. forward_pass( data[ i ]. data );
        if( first_guess != data[ i ]. label ) err ++;
    }
    free( data );
    
    printf("Error Rate = %f%% (%d out of 10,000)\n", double(err)/double(num)*100.0, err);
    printf("Accuracy = %.3f%% (%d out of 10,000)\n", 100.0 - double(err)/double(num)*100.0, num - err);
    return 0;
}