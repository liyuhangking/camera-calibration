#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    
    ifstream fin("calibration.txt");            //读取标定图片的路径
    if (!fin)                                  //检测是否读取到文件
    {
        cerr<<"没有找到文件"<<endl;
    }
    ofstream fout("calibration_result.txt");   //输出结果保存在此文本文件下
    //依次读取每一幅图片，从中提取角点
    cout<<"开始提取角点……"<<endl;
    int image_count = 0;                       //图片数量
    Size image_size;                           //图片尺寸
    Size board_size = Size(7,7);               //标定板每行每列角点个数，共7*7个角点
    vector<Point2f> image_points_buf;          //缓存每幅图检测到的角点
    vector<vector<Point2f>> image_points_seq;  //用一个二维数组保存检测到的所有角点
    string filename;                           //申明一个文件名的字符串
    
    while (getline(fin,filename))              //逐行读取，将行读入字符串   
    {
        image_count++;
        cout<<"image_count = "<<image_count<<endl;
        //读入图片
        Mat imageInput=imread(filename);
        if(image_count == 1)
        {
            image_size.height = imageInput.rows;//图像的高对应着行数
            image_size.width = imageInput.cols; //图像的宽对应着列数
            cout<<"image_size.width = "<<image_size.width<<endl;
            cout<<"image_size.height = "<<image_size.height<<endl;
        }
        //角点检测
        if (findChessboardCorners(imageInput, board_size, image_points_buf) == 0)
        {
            cout<<"can not find the corners "<<endl;
            exit(1);
        }
        else
        {
             Mat gray;                     
             cvtColor(imageInput,gray, cv::COLOR_RGB2GRAY);
             //亚像素精确化
             find4QuadCornerSubpix(gray, image_points_buf, Size(5,5));
             image_points_seq.push_back(image_points_buf);//保存亚像素角点
             //在图中画出角点位置
             drawChessboardCorners(gray, board_size, image_points_buf, true);//将角点连线
             namedWindow("Camera calibration", WINDOW_NORMAL);
             imshow("Camera calibration", gray);
             waitKey(100);                         //等待按键输入
        }
    }
    //输出图像数目
    int total = image_points_seq.size();
    cout<<"total = "<<total<<endl;
    int CornerNum = board_size.width*board_size.height;//一幅图片中的角点数
    cout<<"第一副图片的角点数据:"<<endl;
    for (int i=0; i<CornerNum; i++)
    {
        cout<<"x= "<<image_points_seq[0][i].x<<" ";
        cout<<"y= "<<image_points_seq[0][i].y<<" ";
        cout<<endl;
    }
    cout<<"角点提取完成!\n";
    
    //开始相机标定
    cout<<"开始标定……"<<endl;
    Size square_size = Size(10,10);              //每个小方格实际大小
    vector<vector<Point3f>> object_points;         //保存角点的三维坐标
    Mat cameraMatrix = Mat(3,3,CV_32FC1,Scalar::all(0));//内参矩阵3*3
    Mat distCoeffs = Mat(1,5,CV_32FC1,Scalar::all(0));//畸变矩阵1*5
    vector<Mat> rotationMat;                       //旋转矩阵
    vector<Mat> translationMat;                    //平移矩阵
    //初始化角点三维坐标
    int i,j,t;
    for (t=0; t<image_count; t++)
    {
        vector<Point3f> tempPointSet;
        for (i=0; i<board_size.height; i++)       //行
        {
            for (j=0;j<board_size.width;j++)      //列
            {
                Point3f realpoint;
                realpoint.x = i*square_size.width;
                realpoint.y = j*square_size.height;
                realpoint.z = 0;
                tempPointSet.push_back(realpoint);
            }
        }
        object_points.push_back(tempPointSet);
    }
    vector<int> point_counts;
    for (i=0; i<image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}
    //标定
    calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs,rotationMat, translationMat,0);   //标定函数
    cout<<"标定完成！"<<endl;
  
    //将标定结果写入txt文件
    cout<<"开始保存结果……"<<endl;
    Mat rotate_Mat = Mat(3,3,CV_32FC1, Scalar::all(0));//保存旋转矩阵
    fout<<"相机内参数矩阵："<<endl;
    fout<<cameraMatrix<<endl<<endl;
    fout<<"畸变系数：\n";   
	fout<<distCoeffs<<endl<<endl<<endl; 
    for (int i=0; i<image_count; i++)
    {
        Rodrigues(rotationMat[i], rotate_Mat); //将旋转向量通过公式转换为旋转矩阵
        fout<<"第"<<i+1<<"幅图像的旋转矩阵为："<<endl;
        fout<<rotate_Mat<<endl;
        fout<<"第"<<i+1<<"幅图像的平移向量为："<<endl;
        fout<<translationMat[i]<<endl<<endl;
    }
    cout<<"保存完成"<<endl;
    fout<<endl;
    
    return 0;
}


