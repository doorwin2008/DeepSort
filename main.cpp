#include <fstream>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "inference.h"
#include "tracker.h"//deepsort
#include <unordered_map>
#include "CThread.h"

using namespace std;
using namespace cv;

//-----------------------------------thread
CMutex g_metux;

class canmessageThread : public CThread
{
public:
    canmessageThread(const std::string& strName)
        : m_strThreadName2(strName)
    {
    }

    ~canmessageThread()
    {
    }

public:
    virtual void Run()
    {
        cout << "can message transfer thread" << endl;
    }
private:
    std::string m_strThreadName2;
};

class TestThread : public CThread
{
public:
    TestThread(const std::string& strName)
        : m_strThreadName(strName)
    {
    }

    ~TestThread()
    {
    }

public:
    virtual void Run()
    {
        for (int i = 0; i < 10; i++)
        {
            CLock lock(g_metux);
            std::cout << m_strThreadName << ":" << i << std::endl;
            Sleep(100);
        }
    }
private:
    std::string m_strThreadName;
};

//-----------------------------------thread
const int g_nn_budget=100; //NearestNeighborDistanceMetric参数值
const float g_max_cosine_distance=0.1;//最大余弦距离
unsigned int g_framcount = 0;
double g_frame_height  ;
double g_frame_width  ;

// 初始化存储坐标的字典
struct doublePoint {
    float tx; //transformed x
    float ty;
    int rx; // real x
    int ry;
};

std::unordered_map<int, std::deque<doublePoint>> g_coordinates;
//---------------------
double g_fps;

// 模拟 Ultralytics 检测结果
struct Detection {
    int tracker_id;
    cv::Point2f bottom_center;
};

//坐标转换
//cv::Point2f src_points[] = {
//           cv::Point2f(0,  359),//A
//           cv::Point2f(639,  359),//B
//           cv::Point2f(200,  160),//C
//           cv::Point2f(400,  160) };//D
//cv::Point2f dst_points[] = {
//            cv::Point2f(0,  359),//A
//            cv::Point2f(639,  359),//B
//            cv::Point2f(0,  70),//C'
//            cv::Point2f(639, 70) };//D'
//Mat g_transform;
//坐标转换
//cv::Point2f src_points[] = {
//           cv::Point2f(230,  475),//A
//           cv::Point2f(603,  475),//B
//           //cv::Point2f(313,  423),//C
//           cv::Point2f(333,  410),//C
//           //cv::Point2f(545,  414) };//D
//           cv::Point2f(535,  410) };//D
//cv::Point2f dst_points[] = {
//            cv::Point2f(230,  475),//A
//            cv::Point2f(603,  475),//B
//            cv::Point2f(230,  1),//C'
//            cv::Point2f(600, 1) };//D'
//cv::Point2f src_points[] = {
//           cv::Point2f(230,  475),//A
//           cv::Point2f(603,  475),//B
//           //cv::Point2f(313,  423),//C
//           cv::Point2f(333,  410),//C
//           //cv::Point2f(545,  414) };//D
//           cv::Point2f(535,  410) };//D
//cv::Point2f dst_points[] = {
//            cv::Point2f(230,  475),//A
//            cv::Point2f(603,  475),//B
//            cv::Point2f(230,  1),//C'
//            cv::Point2f(600, 1) };//D'

cv::Point2f src_points[] = {
           cv::Point2f(0,  720-1),//A
           cv::Point2f(1280-1,  720-1),//B

           cv::Point2f(370,  581),//C
           cv::Point2f(1030,  581) };//D
cv::Point2f dst_points[] = {
            cv::Point2f(0,  720 - 1),//A
            cv::Point2f(1280 - 1,  720 - 1),//B
            cv::Point2f(0,  0 + 50),//C'
            cv::Point2f(1280 - 1, 0 + 50) };//D'

Mat g_transform;



// 模拟 ViewTransformer 转换点
std::vector<cv::Point2f> view_transformer_transform_points(const std::vector<cv::Point2f>& points) {
    std::vector<cv::Point2f> transformed_points;
    perspectiveTransform(points, transformed_points, g_transform);
    return transformed_points;
}
void get_detections(DETECTBOX box,float confidence,int classId,DETECTIONS& d)
{
    DETECTION_ROW tmpRow;
    tmpRow.tlwh = box;//DETECTBOX(x, y, w, h);

    tmpRow.confidence = confidence;
    tmpRow.classId = classId;
    d.push_back(tmpRow);
}

int ReadCocoYaml(YOLO_V8*& p) {
    // Open the YAML file
    std::ifstream file("../coco.yaml");
    if (!file.is_open())
    {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    // Read the file line by line
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line))
    {
        lines.push_back(line);
    }

    // Find the start and end of the names section
    std::size_t start = 0;
    std::size_t end = 0;
    for (std::size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i].find("names:") != std::string::npos)
        {
            start = i + 1;
        }
        else if (start > 0 && lines[i].find(':') == std::string::npos)
        {
            end = i;
            break;
        }
    }

    // Extract the names
    std::vector<std::string> names;
    for (std::size_t i = start; i < end; i++)
    {
        std::stringstream ss(lines[i]);
        std::string name;
        std::getline(ss, name, ':'); // Extract the number before the delimiter
        std::getline(ss, name); // Extract the string after the delimiter
        names.push_back(name);
    }
    p->classes = names;
    return 0;
}

//初始化Yolov8 Onnx目标识别 
YOLO_V8* Yolov8DetectInit()
{
    YOLO_V8* yoloDetector = new YOLO_V8;
    ReadCocoYaml(yoloDetector);
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.3;
    params.iouThreshold = 0.5;
    params.modelPath = "D:/doorw_source/yolo8/ultralytics/yolov8n.onnx";
    params.imgSize = { 640, 640 };
#ifdef USE_CUDA
    params.cudaEnable = true;

    // GPU FP32 inference
    params.modelType = YOLO_DETECT_V8;
    // GPU FP16 inference
    //Note: change fp16 onnx model
    //params.modelType = YOLO_DETECT_V8_HALF;
#else
    // CPU inference
    params.modelType = YOLO_DETECT_V8;
    params.cudaEnable = false;
#endif
    yoloDetector->CreateSession(params);
    //Detector(yoloDetector);

    std::cout << "Yolov8 detector init done:" << yoloDetector->test(2) << std::endl;
    return yoloDetector;
}
void deepsortObjDectTrack(cv::Mat& frame, tracker& mytracker, YOLO_V8*& p)
{
    DETECTIONS detections;
    std::vector<DL_RESULT> res;//doorwin
    //YOLO_V8目标识别
    p->RunSession(frame, res);

    for (DL_RESULT re : res){
        if (re.classId == 0 || re.classId == 1 || re.classId == 2 || re.classId == 3 || re.classId == 4 || re.classId == 5 ||  re.classId == 7) { //检测的目标类型
            cv::rectangle(frame, re.box, cv::Scalar(255, 0, 0), 1);
            get_detections(DETECTBOX(re.box.x, re.box.y, re.box.width, re.box.height), re.confidence,re.classId , detections);//给detections赋值
        }
    }

    for (auto& re : res){
        cv::RNG rng(cv::getTickCount());
        //cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        cv::Scalar color(128, 128, 128);
        cv::rectangle(frame, re.box, color, 1);

        float confidence = floor(100 * re.confidence) / 100;
        std::string label = p->classes[re.classId] + " " +
            std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

        //cv::rectangle(frame, cv::Point(re.box.x, re.box.y - 25), cv::Point(re.box.x + label.length() * 10, re.box.y), color, cv::FILLED);
        //显示识别目标的标签，分类+ 权重
        cv::putText(frame, label, cv::Point(re.box.x, re.box.y - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 0);
    }
    
    std::vector<RESULT_DATA> result;
    if(detections.size() >0)
    {
        // std::cout << "get feature succeed!"<<std::endl;
        unsigned int aa = g_framcount ++% 5;

        //if ( aa == 0)
        {
            mytracker.predict(); // 追踪目标预测
            mytracker.update(detections); //追踪目标 更新@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

            // 获取底部中心点坐标
            std::vector<cv::Point2f> BCpoints;
            std::vector<Detection> detectionsspeed;
            std::vector<cv::Rect> detectionXYWH;
            
            //=======================================
            for (Track& track : mytracker.tracks) {
                if (!track.is_confirmed() || track.time_since_update > 1) {
                    //std::cout << "-----" << track.is_confirmed()<<"--"<< track.time_since_update << std::endl;
                    continue;
                }
                result.push_back(std::make_pair(track.track_id, track.to_tlwh()));

            }
            for (unsigned int k = 0; k < detections.size(); k++)
            {
                DETECTBOX tmpbox = detections[k].tlwh;
                cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
                cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 1);
                // cvScalar的储存顺序是B-G-R，CV_RGB的储存顺序是R-G-B   
            }
            for (unsigned int k = 0; k < result.size(); k++)
            {
                DETECTBOX tmp = result[k].second;
                cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
                //rectangle(frame, rect, cv::Scalar(255, 255, 0), 1);

                std::string label = cv::format("ID=%d", result[k].first);
                //速度计算
                Detection det;
                det.tracker_id = result[k].first;
                det.bottom_center = cv::Point2f(rect.x + rect.width/2.0, rect.y + rect.height );
                //cout << "坐标输出：" << rect.x << "-" << rect.y << "-"<< rect.width <<"-"<< rect.height << endl;
                BCpoints.push_back(det.bottom_center);
                detectionsspeed.push_back(det);
                detectionXYWH.push_back(rect) ; //保留原始的坐标数据


                //显示追踪目标ID
                cv::putText(frame, label, cv::Point(rect.x, rect.y + 20), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 2, 0);
            }
            if (result.size() > 0) {

            }
            
            // 坐标转换
            if (BCpoints.size() > 0) {
                BCpoints.push_back(cv::Point2f(g_frame_width/2.0, g_frame_height));//做坐标原点添加到数组中最后一个元素一起转换
                std::vector<cv::Point2f> transformed_points = view_transformer_transform_points(BCpoints);
                // 存储转换后的坐标
                doublePoint doublepoint;//自定义双点坐标数据结构，存储原始坐标和转换后的坐标

                for (size_t i = 0; i < detectionsspeed.size(); ++i) {
                    int tracker_id = detectionsspeed[i].tracker_id;
                    doublepoint.tx =  -(transformed_points[i].y - transformed_points[transformed_points.size()-1].y);
                    doublepoint.ty =-(transformed_points[i].x - transformed_points[transformed_points.size()-1].x);
                    doublepoint.rx = BCpoints[i].x;
                    doublepoint.ry = BCpoints[i].y;
                    
                    //打印图像坐标和宽度高度
                    //cout << "id:" << tracker_id << endl;
                    //cout << "x:" << detectionXYWH[i].x << " y:" << detectionXYWH[i].y << " w:" << detectionXYWH[i].width << " h:" << detectionXYWH[i].height << endl;
                    
                    //实时计算距离
                    doublepoint.tx =doublepoint.tx  /300 ;
                    doublepoint.ty =  doublepoint.ty / 200 ;
 					char buffer[50];

                    sprintf(buffer, "[%.1f , %.1f]", doublepoint.tx, doublepoint.ty);
                   cv::putText(frame, /*" dx:" + std::to_string((int)dx) +*/
                       buffer, cv::Point(doublepoint.rx - 30, doublepoint.ry-18 ), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255), 2, 0);
                    
                    //计算速度
                    if (g_coordinates[tracker_id].size() >= g_fps) {//当长度超过30，删除第一个元素
                        g_coordinates[tracker_id].pop_front();
                    }
                    g_coordinates[tracker_id].push_back(doublepoint);//后面插入一个新的元素
                    //std::cout << "dy: " << dy << std::endl;
                }

                // 计算速度
                for (const auto& det : detectionsspeed) {
                    int tracker_id = det.tracker_id;
                    cout << "id:" << tracker_id << endl;
                    if (g_coordinates[tracker_id].size() > g_fps / 2) {
                        int coordinate_start; //   = g_coordinates[tracker_id].back().ty;//std::deque的最后一个元素
                        int coordinate_end;  //   = g_coordinates[tracker_id].front().ty;//队列的第一个元素
                        int sizeofg_coordinates = g_coordinates[tracker_id].size();
                        if (sizeofg_coordinates > 6) {
                             coordinate_start=
                                ( g_coordinates[tracker_id][sizeofg_coordinates - 1].ty 
                                + g_coordinates[tracker_id][sizeofg_coordinates - 2].ty
                                + g_coordinates[tracker_id][sizeofg_coordinates - 3].ty)/3.0;
                            coordinate_end =
                                ( g_coordinates[tracker_id][0].ty 
                                + g_coordinates[tracker_id][1].ty
                                + g_coordinates[tracker_id][2].ty)/3.0;
                        }
                        int longitudinal_distance = coordinate_start - coordinate_end;
                        int rx = g_coordinates[tracker_id].front().rx ;
                        int ry = g_coordinates[tracker_id].back().ry;
                        double time = static_cast<double>(g_coordinates[tracker_id].size()) / g_fps;
                        double speed = static_cast<double>(longitudinal_distance) / time * 3.6;
                        //std::cout << "Tracker ID: " << tracker_id << ", Speed: " << speed << " km/h" << std::endl;
                       // cv::putText(frame,  std::to_string((int)speed) +" kph" ,cv::Point(g_coordinates[tracker_id].back().rx - 20, g_coordinates[tracker_id].back().ry + 20), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 255), 1, 0); //车速计算
                        //cv::putText(frame, " x:"+std::to_string((int)rx)+
                        //    " y:" + std::to_string((int)ry) ,
                        //    cv::Point(g_coordinates[tracker_id].front().rx-20, g_coordinates[tracker_id].front().ry + 20), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 255), 1, 0);
                    }
                }
            }  
        }
    }
}

int main(int argc, char *argv[])
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR); //只打印错误信息
    CV_LOG_INFO(NULL, "Entering someFunction");
    CV_LOG_ERROR(NULL, "cv log test");
    //deepsort
    tracker mytracker(g_max_cosine_distance, g_nn_budget);
    //坐标转换初始化
    g_transform = getPerspectiveTransform(src_points, dst_points);
    //imgTransform();

    //const char* appname = "Demo";
    //TCANFD tcanfd[100];

    //if (initialize_lib_tsmaster(appname) == 0)
    //    printf("初始化成功\n");
    //else cout << "初始化失败" << endl;
    ////if (tsapp_set_can_channel_count(1) == 0)
    //    printf("通道设置成功\n");

    YOLO_V8* yoloDetectorPr;//YOLOV8目标检测指针
    //-----------------------------------------------------------------------
    // 加载类别名称
    std::vector<std::string> classes;
    std::string file="../coco_80_labels_list.txt";
    std::ifstream ifs(file);
    if (!ifs.is_open()) 
        CV_Error(cv::Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
    std::cout<<"classes:"<<classes.size();
    std::cout<<"begin read video"<<std::endl;

    std::string videoPathName = "D:/doorw_source/yolo8/ultralytics/dvr7.mp4";
     // std::string videoPathName ="D:/doorw_source/yolo8/ultralytics/gaosu-s.mp4";//高架桥固定拍摄车流
    // std::string videoPathName ="D:/doorw_source/yolo8/ultralytics/3.mp4"; //道路行驶
   //std::string videoPathName ="D:/doorw_source/yolo8/ultralytics/2.mp4";//市内高架
   //std::string videoPathName ="D:/doorw_source/yolo8/ultralytics/1.mp4";//市内高架  标定参考
    //std::string videoPathName ="D:/doorw_source/yolo8/ultralytics/football.mp4";
    //std::string videoPathName ="D:/doorw_source/yolo8/ultralytics/tokyo-street.mp4"; 
   //std::string videoPathName ="D:/doorw_source/yolo8/ultralytics/road2.mp4"; 
   // std::string videoPathName ="D:/doorw_source/yolo8/ultralytics/dvr1.mp4";
    //std::string videoPathName ="D:/doorw_source/yolo8/ultralytics/dvr_driver.mp4";
      //std::string videoPathName ="D:/doorw_source/yolo8/ultralytics/dvr_4.mp4";
       //std::string videoPathName ="D:/doorw_source/yolo8/ultralytics/dvr5.mp4";
     //std::string videoPathName ="D:/doorw_source/yolo8/ultralytics/dvr_long_480.mp4";
     // std::string videoPathName ="D:/doorw_source/yolo8/ultralytics/dvr6.mp4";

      //std::string videoPathName ="D:/doorw_source/yolo8/ultralytics/dvr8p.mp4";

    cv::VideoCapture capture(videoPathName);
    //cv::VideoCapture capture(0); std::string videoPathName = "camera0";

    if (!capture.isOpened()) {
        printf("could not read this video file...\n");
        return -1;
    }
    std::cout<<"end read video"<<std::endl;
    int num_frames = 0;
    yoloDetectorPr = Yolov8DetectInit();// Yolo检测初始化
    //读取视频帧率
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    g_fps = capture.get(cv::CAP_PROP_FPS);
    g_frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    g_frame_width  = capture.get(cv::CAP_PROP_FRAME_WIDTH);

    //配置视频保存
    unsigned int fps = capture.get(CAP_PROP_FPS);
    Size frameSize(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT));
    // 创建 VideoWriter 对象
    VideoWriter writer("output.mp4", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frameSize);
    bool videoSave = false;

    //capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    //capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    //设置帧速率为 15 fps
    //capture.set(cv.CAP_PROP_FPS, 15)
    
    //TestThread thread2("Thread2");
    //thread2.Start();
    //canmessageThread canthread("canthread");
    //canthread.Start();

    std::chrono::time_point<std::chrono::steady_clock> startTime = std::chrono::steady_clock::now();
    std::chrono::time_point<std::chrono::steady_clock> endTime;
 
    unsigned int frame_count = 0;
    //unsigned int fps = 0;
    //逐帧检测开始
    while (true)
    {
        cv::Mat frame;

        if (!capture.read(frame)) // if not success, break loop
        {
            std::cout<<"\n video play end.\n";
            break;
        }

        num_frames ++;
        std::cout <<  "-------------frames:" << num_frames << std::endl;

        //Second/Millisecond/Microsecond  秒s/毫秒ms/微秒us
       
        //开始检测目标 doorwin
        deepsortObjDectTrack(frame, mytracker, yoloDetectorPr);//YOLOV8  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        frame_count++;
        endTime = std::chrono::steady_clock::now();
        //auto detect_time =std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();//ms
        double timeTaken = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        if (timeTaken >= 1000)
        {
            fps = frame_count;
            startTime = std::chrono::steady_clock::now();
            frame_count = 0;
        }
                                                                                                     //std::cout<<classes.size()<<":"<<":"<<num_frames<<std::endl;
        std::string label = "Fram Rate:" + std::to_string(fps);

        cv::putText(frame, label, cv::Point(10, 20), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 2, 0);

        cv::imshow("Tracker", frame);

        if (cv::waitKey(1) == 27) // Wait for 'esc' key press to exit
        {
            break;
        }
        if(cv::waitKey(1) == 50) // Wait for 'esc' key press to exit
        {
            videoSave = true;
        }
        if (videoSave)
        {
            // 将帧写入输出视频文件
            writer.write(frame);
        }
        
        if (cv::waitKey(1) == 49)//截图数字1，长按
        {
            // 获取不带路径的文件名
            std::string::size_type iPos = videoPathName.find_last_of('/') + 1;
            std::string filename = videoPathName.substr(iPos, videoPathName.length() - iPos);
            std::cout << "文件名: " << filename << std::endl;

            // 获取不带后缀的文件名
            std::string name = filename.substr(0, filename.rfind("."));
            std::cout << "不带后缀的文件名: " << name << std::endl;

            // 获取后缀名
            std::string suffix_str = filename.substr(filename.find_last_of('.') + 1);
            std::cout << "后缀名: " << suffix_str << std::endl;

            std::string savename = "../images/" + name + std::to_string(num_frames) + ".png";
            cv::imwrite(savename, frame);

            std::cout << "key press:" << cv::waitKey(1) << std::endl;
        }
    }
    capture.release();
    writer.release();
    delete yoloDetectorPr;
    cv::destroyAllWindows();
}
