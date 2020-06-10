#include <openpose/gui/guiInfoAdder.hpp>
#include <cstdio> // std::snprintf
#include <limits> // std::numeric_limits
#include <openpose/utilities/fastMath.hpp>
#include <openpose_private/utilities/openCvPrivate.hpp>

#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include <time.h>

#include <hand/renderHand.cpp>


using namespace std;
using namespace cv;

namespace op
{
    // Used colors
    const cv::Scalar WHITE_SCALAR{255, 255, 255};

    int count = 0;

    void updateFps(unsigned long long& lastId, double& fps, unsigned int& fpsCounter,
                   std::queue<std::chrono::high_resolution_clock::time_point>& fpsQueue,
                   const unsigned long long id, const int numberGpus)
    {
        try
        {
            // If only 1 GPU -> update fps every frame.
            // If > 1 GPU:
                // We updated fps every (3*numberGpus) frames. This is due to the variability introduced by
                // using > 1 GPU at the same time.
                // However, we update every frame during the first few frames to have an initial estimator.
            // In any of the previous cases, the fps value is estimated during the last several frames.
            // In this way, a sudden fps drop will be quickly visually identified.
            if (lastId != id)
            {
                lastId = id;
                fpsQueue.emplace(std::chrono::high_resolution_clock::now());
                bool updatePrintedFps = true;
                if (fpsQueue.size() > 5)
                {
                    const auto factor = (numberGpus > 1 ? 25u : 15u);
                    updatePrintedFps = (fpsCounter % factor == 0);
                    // updatePrintedFps = (numberGpus == 1 ? true : fpsCounter % (3*numberGpus) == 0);
                    fpsCounter++;
                    if (fpsQueue.size() > factor)
                        fpsQueue.pop();
                }
                if (updatePrintedFps)
                {
                    const auto timeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                        fpsQueue.back()-fpsQueue.front()
                    ).count() * 1e-9;
                    fps = (fpsQueue.size()-1) / (timeSec != 0. ? timeSec : 1.);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void addPeopleIds(
        cv::Mat& cvOutputData, const Array<long long>& poseIds, const Array<float>& poseKeypoints,
        const int borderMargin)
    {
        try
        {
            if (!poseIds.empty())
            {
                const auto poseKeypointsArea = poseKeypoints.getSize(1)*poseKeypoints.getSize(2);
                const auto isVisible = 0.05f;

                std::ofstream file;
                count = count +1 ;
                for (auto i = 0u ; i < poseIds.getVolume() ; i++)
                {

                   if (file.bad())
                   {
                       std::cout << "cannot open file" << std::endl;;
                   }

                   file.open("/home/ldb/projects/openpose-master/output/mytest.txt", std::ios::app);


                    //利用ofstream类的构造函数创建一个文件输出流对象来打开文件
                    //ofstream fout( "/home/ldb/projects/openpose-master/output/mytest.txt" );
                    file << "打开文件：第" + std::to_string(count) + "次\n" << endl;
                    for(auto j=0;j< 75;j++)
                    {
                       if (j > 35)
                       {

//                         putTextOnCvMat(cvOutputData, "poseKeypoints =" + std::to_string(poseKeypoints[j]) +"j =" + std::to_string(j),
//                           {600, 20*j - 700}, WHITE_SCALAR, false, cvOutputData.cols);


                         if (!file)
                         {
                           cout << "文件不能打开" <<endl;
                         }
                         else
                         {
                             if(j == 12)
                             {

                               putTextOnCvMat(cvOutputData, "right hand:",
                               {int(poseKeypoints[j]), int(poseKeypoints[j+1])}, WHITE_SCALAR, false, cvOutputData.cols);

                             }
                             else if(j == 21)
                             {
                               putTextOnCvMat(cvOutputData, "left hand:",
                               {int(poseKeypoints[j]), int(poseKeypoints[j+1])}, WHITE_SCALAR, false, cvOutputData.cols);

                             }


                             // 输出到磁盘文件
                             //file << "poseKeypoints =" + std::to_string(poseKeypoints[j]) +"   j =" + std::to_string(j) +"\n"<< endl;

                         }

                       }
                       else
                       {
//                          putTextOnCvMat(cvOutputData, "poseKeypoints --" + std::to_string(poseKeypoints[j]) +"j =" + std::to_string(j),
//                            {10, 20*j+10}, WHITE_SCALAR, false, cvOutputData.cols);


                          if (!file)
                          {
                            cout << "文件不能打开" <<endl;
                          }
                          else
                          {

//                            putTextOnCvMat(cvOutputData, "poseKeypoints =" + std::to_string(poseKeypoints[j]) +"j =" + std::to_string(j),
//                            {600, 20*j - 700}, WHITE_SCALAR, false, cvOutputData.cols);

                                 if(j == 12)
                                 {

//                                   putTextOnCvMat(cvOutputData, "right hand:",
//                                   {int(poseKeypoints[j]), int(poseKeypoints[j+1])}, WHITE_SCALAR, false, cvOutputData.cols);


                                   if(int(poseKeypoints[j]) -220 >0 && int(poseKeypoints[j+1]) - 190 >0
                                   && int(poseKeypoints[j]) + 210 < cvOutputData.cols && int(poseKeypoints[j+1]) + 230 < cvOutputData.rows )
                                   {

                                       Rect rect(int(poseKeypoints[j]) -220, int(poseKeypoints[j+1]) - 190, 430, 420); //左上坐标（x,y）和矩形的长(x)宽(y)
//                                       Scalar color = Scalar(0,255,0);
//                                       cv::rectangle(cvOutputData, rect, Scalar(0, 255, 0),1, LINE_8,0);
                                       time_t timep;
                                       struct tm *p;
                                       time(&timep); //获取从1970至今过了多少秒，存入time_t类型的timep
                                       p = localtime(&timep);//用localtime将秒数转化为struct tm结构体
                                       string imgName = "rightHand" +std::to_string(1900 + p->tm_year) + std::to_string(1+ p->tm_mon) + std::to_string(p->tm_mday)
                                       + std::to_string(p->tm_hour) + std::to_string(p->tm_min) + std::to_string(p->tm_sec);

                                       string imgPath = "/home/ldb/projects/openpose-master/output/partImg/" + imgName + ".jpg";
                                       cv::imwrite(imgPath,cvOutputData(rect));

                                   }



                                 }
                                 else if(j == 21)
                                 {
                                   putTextOnCvMat(cvOutputData, "left hand:",
                                   {int(poseKeypoints[j]), int(poseKeypoints[j+1])}, WHITE_SCALAR, false, cvOutputData.cols);


                                   if(int(poseKeypoints[j]) - 250 >0 && int(poseKeypoints[j+1]) - 180 >0
                                   && int(poseKeypoints[j]) + 190 < cvOutputData.cols && int(poseKeypoints[j+1]) + 190 < cvOutputData.rows )
                                   {

                                       Rect rect(int(poseKeypoints[j]) -250, int(poseKeypoints[j+1]) - 180, 440, 370); //左上坐标（x,y）和矩形的长(x)宽(y)
//                                       Scalar color = Scalar(0,255,0);
//                                       cv::rectangle(cvOutputData, rect, Scalar(0, 255, 0),1, LINE_8,0);
                                       time_t timep;
                                       struct tm *p;
                                       time(&timep); //获取从1970至今过了多少秒，存入time_t类型的timep
                                       p = localtime(&timep);//用localtime将秒数转化为struct tm结构体
                                       string imgName = "leftHand" +std::to_string(1900 + p->tm_year) + std::to_string(1+ p->tm_mon) + std::to_string(p->tm_mday)
                                       + std::to_string(p->tm_hour) + std::to_string(p->tm_min) + std::to_string(p->tm_sec);

                                       string imgPath = "/home/ldb/projects/openpose-master/output/partImg/" + imgName + ".jpg";
                                       //cv::imwrite(imgPath,cvOutputData(rect));

                                   }

                                 }

                                 // 输出到磁盘文件
                                 //file << "poseKeypoints =" + std::to_string(poseKeypoints[j]) +"   j =" + std::to_string(j) +"\n"<< endl;

                          }

                       }

                    }
                    file.close();


                    if (poseIds[i] > -1)
                    {
                        const auto indexMain = i * poseKeypointsArea;
                        const auto indexSecondary = i * poseKeypointsArea + poseKeypoints.getSize(2);
                        if (poseKeypoints[indexMain+2] > isVisible || poseKeypoints[indexSecondary+2] > isVisible)
                        {
                            const auto xA = positiveIntRound(poseKeypoints[indexMain]);
                            const auto yA = positiveIntRound(poseKeypoints[indexMain+1]);
                            const auto xB = positiveIntRound(poseKeypoints[indexSecondary]);
                            const auto yB = positiveIntRound(poseKeypoints[indexSecondary+1]);
                            int x;
                            int y;
                            if (poseKeypoints[indexMain+2] > isVisible && poseKeypoints[indexSecondary+2] > isVisible)
                            {
                                const auto keypointRatio = positiveIntRound(
                                    0.15f * std::sqrt((xA-xB)*(xA-xB) + (yA-yB)*(yA-yB)));
                                x = xA + 3*keypointRatio;
                                y = yA - 3*keypointRatio;
                            }
                            else if (poseKeypoints[indexMain+2] > isVisible)
                            {
                                x = xA + positiveIntRound(0.25f*borderMargin);
                                y = yA - positiveIntRound(0.25f*borderMargin);
                            }
                            else //if (poseKeypoints[indexSecondary+2] > isVisible)
                            {
                                x = xB + positiveIntRound(0.25f*borderMargin);
                                y = yB - positiveIntRound(0.5f*borderMargin);
                            }
                            putTextOnCvMat(cvOutputData, "poseIds" + std::to_string(poseIds[i]), {x, y}, WHITE_SCALAR, false, cvOutputData.cols);
                        }
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    GuiInfoAdder::GuiInfoAdder(const int numberGpus, const bool guiEnabled) :
        mNumberGpus{numberGpus},
        mGuiEnabled{guiEnabled},
        mFpsCounter{0u},
        mLastElementRenderedCounter{std::numeric_limits<int>::max()},
        mLastId{std::numeric_limits<unsigned long long>::max()}
    {
    }

    GuiInfoAdder::~GuiInfoAdder()
    {
    }

    void GuiInfoAdder::addInfo(Matrix& outputData, const int numberPeople, const unsigned long long id,
                               const std::string& elementRenderedName, const unsigned long long frameNumber,
                               const Array<long long>& poseIds, const Array<float>& poseKeypoints)
    {
        try
        {
            cv::Mat cvOutputData = OP_OP2CVMAT(outputData);
            // Sanity check
            if (cvOutputData.empty())
                error("Wrong input element (empty outputData).", __LINE__, __FUNCTION__, __FILE__);
            // Size
            const auto borderMargin = positiveIntRound(fastMax(cvOutputData.cols, cvOutputData.rows) * 0.025);
            // Update fps
            updateFps(mLastId, mFps, mFpsCounter, mFpsQueue, id, mNumberGpus);
            // Fps or s/gpu
            char charArrayAux[15];
            std::snprintf(charArrayAux, 15, "%4.1f fps", mFps);
            // Recording inverse: sec/gpu
            // std::snprintf(charArrayAux, 15, "%4.2f s/gpu", (mFps != 0. ? mNumberGpus/mFps : 0.));
            putTextOnCvMat(
                cvOutputData, charArrayAux, {positiveIntRound(cvOutputData.cols - borderMargin), borderMargin},
                WHITE_SCALAR, true, cvOutputData.cols);
            // Part to show
            // Allowing some buffer when changing the part to show (if >= 2 GPUs)
            // I.e. one GPU might return a previous part after the other GPU returns the new desired part, it looks
            // like a mini-bug on screen
            // Difference between Titan X (~110 ms) & 1050 Ti (~290ms)
            if (mNumberGpus == 1 || (elementRenderedName != mLastElementRenderedName
                                     && mLastElementRenderedCounter > 4))
            {
                mLastElementRenderedName = elementRenderedName;
                mLastElementRenderedCounter = 0;
            }
            mLastElementRenderedCounter = fastMin(mLastElementRenderedCounter, std::numeric_limits<int>::max() - 5);
            mLastElementRenderedCounter++;
            // Add each person ID
            addPeopleIds(cvOutputData, poseIds, poseKeypoints, borderMargin);
            // OpenPose name as well as help or part to show
            putTextOnCvMat(cvOutputData, "OpenPose - " +
                           (!mLastElementRenderedName.empty() ?
                                mLastElementRenderedName : (mGuiEnabled ? "'h' for help" : "")),
                           {borderMargin, borderMargin}, WHITE_SCALAR, false, cvOutputData.cols);
            // Frame number
            putTextOnCvMat(cvOutputData, "Frame: " + std::to_string(frameNumber),
                           {borderMargin, (int)(cvOutputData.rows - borderMargin)}, WHITE_SCALAR, false, cvOutputData.cols);
            // Number people
            putTextOnCvMat(cvOutputData, "People: " + std::to_string(numberPeople),
                           {(int)(cvOutputData.cols - borderMargin), (int)(cvOutputData.rows - borderMargin)},
                           WHITE_SCALAR, true, cvOutputData.cols);

            // poseKeypoints
//            putTextOnCvMat(cvOutputData, "poseKeypoints: " + std::to_string(poseKeypoints[5]),
//                           {(int)(cvOutputData.cols - borderMargin - 50), (int)(cvOutputData.rows - borderMargin - 50)},
//                           WHITE_SCALAR, true, cvOutputData.cols);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
