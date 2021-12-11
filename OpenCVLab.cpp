#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

Mat blurIM(Mat &source,const int &koof, const double &sigma) {
    Mat result = source;
    vector<double> weights(koof*koof,0);
    int pointx = -koof;
    int pointy =  koof;
    // calculating 1/(2sigma^2)
    double gausK = 1.00000 / (2.0000 * pow(sigma,2));
    // calculating 1/(2pi*sigma^2)
    double gausK_ = gausK / 3.14;
    double summ = 0;
    // calculating the gausian kernel wih readius koof
    for (int row = 0; row < koof; row++) {
        for (int col = 0; col < koof; col++) {
            weights[row*koof + col] = gausK_ * exp(-(pow(pointx, 2) + pow(pointy, 2))*gausK);
            summ += weights[row * koof + col];
            pointx += 1;
        }
        pointx = -koof;
        pointy -= 1;
    }
    for (int row = 0; row < koof; row++) {
        for (int col = 0; col < koof; col++) {
            weights[row * koof + col] /= summ;
        }
    }
    // bluring image
    std::cout << "Imsize:" << source.rows-koof << " / " << source.cols <<endl;
    for (int row = 0; row < source.rows; row++) {
        for (int col = 0; col < source.cols; col++) {
            // Mat(x,y) .* Gausian kernel (x,y)
            pointx = col-koof;
            pointy = row+koof;
            //std::cout << row-koof << " " << col+koof << endl;

            double Rsumm = 0, Gsumm = 0, Bsumm = 0;
            
            for (int row_ = 0; row_ < koof; row_++) {
                for (int col_ = 0; col_ < koof; col_++) {
                    int pointy_ = pointy;
                    if (pointy >= source.rows) {
                        pointy = source.rows - (pointy - (source.rows - 1));
                    }
                    // B channel pixel revalue
                    Bsumm += weights[row_ * koof + col_] * source.at<Vec3b>(abs(pointy), abs(pointx))[0];
                    // G channel pixel revalue
                    Gsumm += weights[row_ * koof + col_] * source.at<Vec3b>(abs(pointy), abs(pointx))[1];
                    // R channel pixel revalue
                    Rsumm += weights[row_ * koof + col_] * source.at<Vec3b>(abs(pointy), abs(pointx))[2];           
                    if (pointx >= source.cols-1)pointx = pointx--;
                    else pointx++;
                    pointy = pointy_;
                }
               pointy--;
               pointx = col - koof;
            }
            result.at<Vec3b>(row, col)[0] = Bsumm;
            result.at<Vec3b>(row, col)[1] = Gsumm;
            result.at<Vec3b>(row, col)[2] = Rsumm;
        }
    }

    namedWindow("Blured");
    imshow("Blured", result);
    imwrite("Blured_.jpg", result);
    waitKey(0);
    return result;
}

//Gradient function
void Sobel(Mat &source) {
    Mat resultx = source;
    Mat resulty = source;
    Mat resultSumm = source;
    for (int row = 0; row < source.rows; row++) {
        for (int col = 0; col < source.cols-1; col++) {
           //Finding derivatives
            resultx.at<uchar>(row, col) = (source.at<uchar>(row, col + 1) - source.at<uchar>(row, col));          
        }
    }
    namedWindow("X");
    imshow("X", resultx);
    imwrite("X.jpg", resultx);
    waitKey(0);
    //Y dirivative
    for (int row = 0; row < source.rows - 1; row++) {
        for (int col = 0; col < source.cols; col++) {
            resulty.at<uchar>(row, col) = (source.at<uchar>(row + 1, col) - source.at<uchar>(row, col));
    
        }
    }
    namedWindow("Y");
    imshow("Y", resulty);
    imwrite("Y.jpg", resulty );
    waitKey(0);
    //Mixing derivatives
    for (int row = 0; row < source.rows - 1; row++) {
        for (int col = 0; col < source.cols; col++) {
            resultSumm.at<uchar>(row, col) = (resultx.at<uchar>(row , col) + source.at<uchar>(row, col)) ;
        }
    }
    namedWindow("Summ");
    imshow("Z", resultSumm);
    imwrite("Summed.jpg", resultSumm);
    waitKey(0);
    return;
}

int main()
{
    Mat image = imread("Hatkid.jpg", IMREAD_COLOR);
    namedWindow("test1");
    imshow("test1", image);
    image = blurIM(image, 10, 10);
    Mat gray;
    cvtColor(image, gray,COLOR_RGB2GRAY);
    Sobel(gray);
 
    return 0;
}
