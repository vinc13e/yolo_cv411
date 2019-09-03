
// Brief Sample of using OpenCV dnn module in real time with device capture, video and image.
// VIDEO DEMO: https://www.youtube.com/watch?v=NHtRlndE2cg
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <thread>
#include <mutex>

using namespace std;
using namespace cv;
using namespace cv::dnn;
static const char* about =
        "This sample uses You only look once (YOLO)-Detector (https://arxiv.org/abs/1612.08242) to detect objects on camera/video/image.\n"
        "Models can be downloaded here: https://pjreddie.com/darknet/yolo/\n"
        "Default network is 416x416.\n"
        "Class names can be downloaded here: https://github.com/pjreddie/darknet/tree/master/data\n";

static const char* params =
        "{ help           | false | print usage         }"
        "{ cfg            |       | model configuration }"
        "{ model          |       | model weights       }"
        "{ camera_device  | 0     | camera device number}"
        "{ source         |       | video or image for detection}"
        "{ min_confidence | 0.24  | min confidence      }"
        "{ class_names    |       | File with class names, [PATH-TO-DARKNET]/data/coco.names }";

std::mutex  mtx;
void capThread(string filename, cv::Mat & frame){
    VideoCapture cap;
    cap.open(filename);
    cout << "filename: " << filename << endl;
    while(true){
        mtx.lock();
        cap >> frame;
        ///cv::resize(frame, frame, cv::Size(0,0), 2.5,2.5);
        mtx.unlock();
        std::this_thread::sleep_for(25ms);
    }
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, params);
    if (parser.get<bool>("help"))
    {
        cout << about << endl;
        parser.printMessage();
        return 0;
    }
    String modelConfiguration = parser.get<String>("cfg");
    String modelBinary = parser.get<String>("model");
    dnn::Net net = readNetFromDarknet(modelConfiguration, modelBinary);
    if (net.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "cfg-file:     " << modelConfiguration << endl;
        cerr << "weights-file: " << modelBinary << endl;
        cerr << "Models can be downloaded here:" << endl;
        cerr << "https://pjreddie.com/darknet/yolo/" << endl;
        exit(-1);
    }


    Mat _frame;
    auto th = std::thread(capThread,parser.get<String>("source"), std::ref(_frame));
    th.detach();

    std::this_thread::sleep_for(3s);

    vector<string> classNamesVec;
    ifstream classNamesFile(parser.get<String>("class_names").c_str());
    if (classNamesFile.is_open())
    {
        string className = "";
        while (std::getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }
    for(;;)
    {
        mtx.lock();
        Mat frame = _frame.clone();
        mtx.unlock();

        if (frame.empty())
        {
            cout << "no video frame !!";
            std::this_thread::sleep_for(3s);
            waitKey();
            continue;
        }
        if (frame.channels() == 4)
            cvtColor(frame, frame, COLOR_BGRA2BGR);
        Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(123.68,116.77,103.93), true, false); //Convert Mat to batch of images
        net.setInput(inputBlob, "data");                   //set the network input
//        auto layernames = net.getLayerNames();
////        for(auto & l : layernames ) cout << "l " << l << endl;
////        return 0;
//        auto ln = layernames.at(layernames.size()-1);
//        cout << "ln: " << ln << endl;
        Mat detectionMat = net.forward();   //compute output

        static std::vector<int> outLayers = net.getUnconnectedOutLayers();
        static std::string outLayerType = net.getLayer(outLayers[0])->type;
        vector<double> layersTimings;
        double freq = getTickFrequency() / 1000;
        double time = net.getPerfProfile(layersTimings) / freq;
        ostringstream ss;
        ss << "FPS: " << 1000/time << " ; time: " << time << " ms";
        putText(frame, ss.str(), Point(20,20), 0, 0.5, Scalar(0,0,255));
        float confidenceThreshold = parser.get<float>("min_confidence");


        // Network produces output blob with a shape NxC where N is a number of
        // detected objects and C is a number of classes + 4 where the first 4
        // numbers are [center_x, center_y, width, height]

        for (int i = 0; i < detectionMat.rows; i++)
        {

            const int probability_index = 5;
            const int probability_size = detectionMat.cols - probability_index;
            float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
            size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
            float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

            if (confidence > confidenceThreshold)
            {
                float x = detectionMat.at<float>(i, 0);
                float y = detectionMat.at<float>(i, 1);
                float width = detectionMat.at<float>(i, 2);
                float height = detectionMat.at<float>(i, 3);
                int xLeftBottom = static_cast<int>((x - width / 2) * frame.cols);
                int yLeftBottom = static_cast<int>((y - height / 2) * frame.rows);
                int xRightTop = static_cast<int>((x + width / 2) * frame.cols);
                int yRightTop = static_cast<int>((y + height / 2) * frame.rows);
                Rect object(xLeftBottom, yLeftBottom,
                            xRightTop - xLeftBottom,
                            yRightTop - yLeftBottom);
                rectangle(frame, object, Scalar(0, 255, 0));
                if (objectClass < classNamesVec.size())
                {
                    ss.str("");
                    ss << confidence;
                    String conf(ss.str());
                    String label = String(classNamesVec[objectClass]) + ": " + conf;
                    int baseLine = 0;
                    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                    rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom ),
                                          Size(labelSize.width, labelSize.height + baseLine)),
                              Scalar(255, 255, 255), cv::FILLED);
                    putText(frame, label, Point(xLeftBottom, yLeftBottom+labelSize.height),
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
                }
                else
                {
                    cout << "Class: " << objectClass << endl;
                    cout << "Confidence: " << confidence << endl;
                    cout << " " << xLeftBottom
                         << " " << yLeftBottom
                         << " " << xRightTop
                         << " " << yRightTop << endl;
                }
                cout << "obj " << classNamesVec[objectClass] << " conf " << confidence << std::endl;
            }


        }
        imshow("YOLO: Detections", frame);
        if (waitKey(1) >= 0) break;
    }
    return 0;
} // main

//// ./cv411 -source=rtsp://@81.45.178.30:555/media/video2  -cfg=../data/yolov3.cfg -model=../data/yolov3.weights -class_names=../data/coco.names -min_confidence=0.000