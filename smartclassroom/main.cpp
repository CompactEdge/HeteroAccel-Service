/*
// Copyright (C) 2018-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <iostream>
#include <vector>
#include <string>

#include <monitors/presenter.h>

#include <utils/args_helper.hpp>
#include <utils/images_capture.h>
#include <utils/performance_metrics.hpp>
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>
#include <utils/default_flags.hpp>
#include <unordered_map>
#include <gflags/gflags.h>

#include <pipelines/async_pipeline.h>
#include <pipelines/metadata.h>

#include <models/hpe_model_associative_embedding.h>
#include <models/hpe_model_openpose.h>


extern "C"
{
#include <sys/types.h>
#include <sys/time.h>
#include <sys/queue.h>
#include <stdlib.h>
#include <err.h>
#include <event.h>
#include <evhttp.h>
#include <event2/http_struct.h>
#include <event2/keyvalq_struct.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

#include <cstdio>
#include <sys/stat.h>
#include <cstdlib>
#include <signal.h>

}
//ParseAndCheckCommandLine
DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message.";
static const char at_message[] = "Required. Type of the network, either 'ae' for Associative Embedding, 'higherhrnet' for HigherHRNet models based on ae "
"or 'openpose' for OpenPose.";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char target_size_message[] = "Optional. Target input size.";
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available devices is shown below). "
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
"The demo will look for a suitable plugin for a specified device.";
static const char performance_counter_message[] = "Optional. Enables per-layer performance report.";
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "
"Absolute path to the .xml file with the kernel descriptions.";
static const char custom_cpu_library_message[] = "Required for CPU custom layers. "
"Absolute path to a shared library with the kernel implementations.";
static const char thresh_output_message[] = "Optional. Probability threshold for poses filtering.";
static const char nireq_message[] = "Optional. Number of infer requests. If this option is omitted, number of infer requests is determined automatically.";
static const char num_threads_message[] = "Optional. Number of threads.";
static const char num_streams_message[] = "Optional. Number of streams to use for inference on the CPU or/and GPU in "
"throughput mode (for HETERO and MULTI device cases use format "
"<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";
static const char no_show_message[] = "Optional. Don't show output.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
static const char output_resolution_message[] = "Optional. Specify the maximum output window resolution "
    "in (width x height) format. Example: 1280x720. Input frame size used by default.";

DEFINE_bool(h, false, help_message);
DEFINE_string(at, "", at_message);
DEFINE_string(m, "", model_message);
DEFINE_uint32(tsize, 0, target_size_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_bool(pc, false, performance_counter_message);
DEFINE_string(c, "", custom_cldnn_message);
DEFINE_string(l, "", custom_cpu_library_message);
DEFINE_double(t, 0.1, thresh_output_message);
DEFINE_uint32(nireq, 0, nireq_message);
DEFINE_uint32(nthreads, 0, num_threads_message);
DEFINE_string(nstreams, "", num_streams_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_string(u, "", utilization_monitors_message);
DEFINE_string(output_resolution, "", output_resolution_message);

/**
* \brief This function shows a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "human_pose_estimation_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -at \"<type>\"              " << at_message << std::endl;
    std::cout << "    -i                        " << input_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "    -o \"<path>\"               " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"            " << limit_message << std::endl;
    std::cout << "    -tsize                    " << target_size_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"    " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"    " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"             " << target_device_message << std::endl;
    std::cout << "    -pc                       " << performance_counter_message << std::endl;
    std::cout << "    -t                        " << thresh_output_message << std::endl;
    std::cout << "    -nireq \"<integer>\"        " << nireq_message << std::endl;
    std::cout << "    -nthreads \"<integer>\"     " << num_threads_message << std::endl;
    std::cout << "    -nstreams                 " << num_streams_message << std::endl;
    std::cout << "    -loop                     " << loop_message << std::endl;
    std::cout << "    -no_show                  " << no_show_message << std::endl;
    std::cout << "    -output_resolution        " << output_resolution_message << std::endl;
    std::cout << "    -u                        " << utilization_monitors_message << std::endl;
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_at.empty()) {
        throw std::logic_error("Parameter -at is not set");
    }

    if (!FLAGS_output_resolution.empty() && FLAGS_output_resolution.find("x") == std::string::npos) {
        throw std::logic_error("Correct format of -output_resolution parameter is \"width\"x\"height\".");
    }
    return true;
}






//////
int delay = 33;
double inferenceTime = 0.0;
//std::stringstream rawPose;
int break_point=0;

cv::Mat decodedImage;


// std::stringstream thread_test(cv::Mat decodedImage , HumanPoseEstimator estimator);

//HumanPoseEstimator estimator("\0","\0",false);
void generic_cb(struct evhttp_request *req, void *arg)
{

    /* Response */
    evhttp_send_reply(req, HTTP_NOTFOUND, "Not found", NULL);
}
int findString(char *str, const char *sub)
{
    const char *p1, *p2, *p3;
    int i = 0, j = 0, flag = 0;

    p1 = str;
    p2 = sub;

    for (i = 0; (unsigned int)i < strlen(str); i++)
    {
        if (*p1 == *p2)
        {
            p3 = p1;
            for (j = 0; (unsigned int)j < strlen(sub); j++)
            {
                if (*p3 == *p2)
                {
                    p3++;
                    p2++;
                }
                else
                    break;
            }
            p2 = sub;
            if ((unsigned int)j == strlen(sub))
            {
                flag = 1;
                //printf("\nSubstring found at index : %d\n", i);
            }
        }
        p1++;
    }
    if (flag == 0)
    {
        printf("Substring NOT found");
    }
    return i;
}



void (*breakCapture)(int);

void signalingHandler(int signo) {
  printf("'Ctrl + C' processing...\n");
  break_point=1;

  exit(1);
}
cv::Mat renderHumanPose(HumanPoseResult& result) {
    if (!result.metaData) {
        throw std::invalid_argument("Renderer: metadata is null");
    }

    auto outputImg = result.metaData->asRef<ImageMetaData>().img;

    if (outputImg.empty()) {
        throw std::invalid_argument("Renderer: image provided in metadata is empty");
    }
    //outputTransform.resize(outputImg);
    static const cv::Scalar colors[HPEOpenPose::keypointsNumber] = {
        cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
        cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0),
        cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170),
        cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
        cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
        cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85)
    };
    static const std::pair<int, int> keypointsOP[] = {
        {1, 2}, {1, 5}, {2, 3}, {3, 4},  {5, 6}, {6, 7},
        {1, 8}, {8, 9}, {9, 10}, {1, 11}, {11, 12}, {12, 13},
        {1, 0}, {0, 14},{14, 16}, {0, 15}, {15, 17}
    };
    static const std::pair<int, int> keypointsAE[] = {
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11},
        {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10},
        {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}
    };
    const int stickWidth = 4;
    const cv::Point2f absentKeypoint(-1.0f, -1.0f);
    // for (auto& pose : result.poses) {
    //     for (size_t keypointIdx = 0; keypointIdx < pose.keypoints.size(); keypointIdx++) {
    //         if (pose.keypoints[keypointIdx] !=outFrame absentKeypoint) {
    //             outputTransform.scaleCoord(pose.keypoints[keypointIdx]);
    //             cv::circle(outputImg, pose.keypoints[keypointIdx], 4, colors[keypointIdx], -1);outputImg
    //         }
    //     }
    // }
//--
    std::stringstream rawPose;
    //  for (auto& pose : result.poses) {
       
    //     for (size_t keypointIdx = 0; keypointIdx < pose.keypoints.size(); keypointIdx++) {
    //        //rawPose <<  pose.keypoints[keypointIdx];
    //        //std::cout << pose.keypoints[keypointIdx]<<"x:"<<pose.keypoints[keypointIdx].x<<"y:"<<pose.keypoints[keypointIdx].y<<std::endl;
           
    //     }
    // }
    rawPose << "{\"inferencetime\":" << "inferenceTime" << ","
                        << "\"data\":[";
    for (auto& pose : result.poses) {
        {
        unsigned int id_pose = 0;
        rawPose << "{\"x\": [";
        rawPose << std::fixed << std::setprecision(0);
        int kp_size = pose.keypoints.size();
        int idx = 0;
        for (auto const &keypoint : pose.keypoints)
        {
            if (idx == kp_size - 1)
            {
                rawPose << keypoint.x;
            }
            else
            {
                rawPose << keypoint.x << ",";
            }
            idx++;
        }
        rawPose << "],"
                << "\"y\": [";
        int idy = 0;
        for (auto const &keypoint : pose.keypoints)
        {
            if (idy == kp_size - 1)
            {
                rawPose << keypoint.y;
            }
            else
            {
                rawPose << keypoint.y << ",";
            }
            idy++;
        }
        //rawPose << pose.score;
        rawPose << "]}";
        //rawPose << "\n";
        //std::cout << rawPose.str() << std::endl;
        id_pose++;
        if (id_pose != result.poses.size())
            rawPose << ",";
        }
    rawPose << "]}";
    }
    std::cout << rawPose.str().c_str() << std::endl;
//--
    std::vector<std::pair<int, int>> limbKeypointsIds;
    if (!result.poses.empty()) {
        if (result.poses[0].keypoints.size() == HPEOpenPose::keypointsNumber) {
            limbKeypointsIds.insert(limbKeypointsIds.begin(), std::begin(keypointsOP), std::end(keypointsOP));
        }
        else {
            limbKeypointsIds.insert(limbKeypointsIds.begin(), std::begin(keypointsAE), std::end(keypointsAE));
        }
    }
    cv::Mat pane = outputImg.clone();
    for (auto pose : result.poses) {
        for (const auto& limbKeypointsId : limbKeypointsIds) {
            std::pair<cv::Point2f, cv::Point2f> limbKeypoints(pose.keypoints[limbKeypointsId.first],
                    pose.keypoints[limbKeypointsId.second]);
            if (limbKeypoints.first == absentKeypoint
                    || limbKeypoints.second == absentKeypoint) {
                continue;
            }

            float meanX = (limbKeypoints.first.x + limbKeypoints.second.x) / 2;
            float meanY = (limbKeypoints.first.y + limbKeypoints.second.y) / 2;
            cv::Point difference = limbKeypoints.first - limbKeypoints.second;
            double length = std::sqrt(difference.x * difference.x + difference.y * difference.y);
            int angle = static_cast<int>(std::atan2(difference.y, difference.x) * 180 / CV_PI);
            std::vector<cv::Point> polygon;
            cv::ellipse2Poly(cv::Point2d(meanX, meanY), cv::Size2d(length / 2, stickWidth),
                             angle, 0, 360, 1, polygon);
            cv::fillConvexPoly(pane, polygon, colors[limbKeypointsId.second]);
        }
    }
    cv::addWeighted(outputImg, 0.4, pane, 0.6, 0, outputImg);
    return outputImg;
}


void HttpGenericCallback(struct evhttp_request *req, void *arg)
{
    const char *cmdtype;
    struct evkeyvalq *headers;
    struct evkeyval *header;
    struct evbuffer *buf;

    // cv::Mat decodedImage;

    switch (evhttp_request_get_command(req))
    {
    case EVHTTP_REQ_GET:
        cmdtype = "GET";
        break;
    case EVHTTP_REQ_POST:
        cmdtype = "POST";
        break;
    case EVHTTP_REQ_HEAD:
        cmdtype = "HEAD";
        break;
    case EVHTTP_REQ_PUT:
        cmdtype = "PUT";
        break;
    case EVHTTP_REQ_DELETE:
        cmdtype = "DELETE";
        break;
    case EVHTTP_REQ_OPTIONS:
        cmdtype = "OPTIONS";
        break;
    case EVHTTP_REQ_TRACE:
        cmdtype = "TRACE";
        break;
    case EVHTTP_REQ_CONNECT:
        cmdtype = "CONNECT";
        break;
    case EVHTTP_REQ_PATCH:
        cmdtype = "PATCH";
        break;
    default:
        cmdtype = "unknown";
        break;
    }

    printf("Received a %s request for %s\nHeaders:\n", cmdtype, evhttp_request_get_uri(req));

    headers = evhttp_request_get_input_headers(req);

    for (header = headers->tqh_first; header; header = header->next.tqe_next)
    {
        //    printf("  %s: %s\n", header->key, header->value);
    }
    // printf("http response=%d\n",req->response_code);
    buf = evhttp_request_get_input_buffer(req);
    //printf("buf size %d\n",evbuffer_get_length(buf));

    // printf("len:%zu  body size:%zu\n", evbuffer_get_length(buf), req->body_size);
    // char *tmp = (char*)malloc(evbuffer_get_length(buf)+1);
    // memcpy(tmp, evbuffer_pullup(buf, -1), evbuffer_get_length(buf));
    // tmp[evbuffer_get_length(buf)] = '\0';
    // printf("print the body:\n");
    // printf("HTML BODY:%s\n body end", tmp);
    // free(tmp);
    //puts("input data <<<");
    cv::Mat imgbuf;
    //cv::Mat imgMat;
    //int remaind = 0;
    const char *sub_start = "stream";
    const char *sub_end = "--";
    //char *sub_end = "ho";
    char cbuf[evbuffer_get_length(buf)];
    memset(cbuf, 0, sizeof(char) * evbuffer_get_length(buf));
    int start_idx = 0;
    int end_idx = 0;
    int nbytes = 0;

    while (evbuffer_get_length(buf))
    {
        nbytes = evbuffer_remove(buf, cbuf, sizeof(cbuf));

        if (nbytes > 0)
        {
            // for (int j = 0; j < nbytes; j++)
            // {
            //   printf("%c",cbuf[j]);
            // }

            //(void) fwrite(cbuf, 1, n, stdout);
            int idx_start = findString(cbuf, sub_start);
            int idx_end = findString(cbuf, sub_end);
            //int idx_end = nbytes;
            //printf("idx = %d %d\n",idx_start,idx_end);
            start_idx = idx_start - 1;
            end_idx = idx_end;
        }
        // remaind += nbytes;
    }

    int img_byte_size = nbytes - start_idx - 7;
    printf("data: %d,%d,%d", nbytes, start_idx, end_idx);

    unsigned char img_bytes[img_byte_size];
    memset(img_bytes, 0, sizeof(unsigned char) * img_byte_size);
    memcpy(img_bytes, cbuf + (start_idx - 7), img_byte_size * sizeof(unsigned char));

    cv::Mat rawData(1, img_byte_size, CV_8UC1, (void *)img_bytes);
    decodedImage = imdecode(rawData, cv::IMREAD_COLOR);
    if (decodedImage.data == NULL)
    {
        puts("decodeddataNULL!!! \n");
    }
    

    std::unique_ptr<ResultBase> result;
    std::cout << result << std::endl;
    cv::Mat outFrame = renderHumanPose(result->asRef<HumanPoseResult>());
    // std::stringstream rawPose = thread_test(decodedImage , *estimator);

    static unsigned int count = 1;

    struct evbuffer *buf1 = evbuffer_new();
    //picture_a = decodedImage;
    // if (!buf1 || rawPose.str().empty())
    // {
    //     puts("failed to create response buffer \n");
    //     return;
    // }
    // else
    // {
     
    //     std::cout << rawPose.str().c_str() << std::endl;
    //     // rawPose.str().c_str();
    //     // evbuffer_add_printf(buf1, "%s", rawPose.str().c_str());
    //     // evhttp_send_reply(req, HTTP_OK, "OK", buf1);
    //     // evhttp_send_reply(req, HTTP_OK, "OK", buf1);
    //     evhttp_send_reply(req, 200, "OK", NULL);
    //     // rawPose.str("");
    //     // rawPose.clear();
    //     evbuffer_free(buf1);
    // }

    printf("\n------------------%u------------------\n", count++);
}


void test(int argc, char *argv[]) {
    try {
        PerformanceMetrics metrics;
        
        slog::info << "InferenceEngine: " << printable(*InferenceEngine::GetInferenceEngineVersion()) << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        // if (!ParseAndCheckCommandLine(argc, argv)) {
        //     return 0;
        // }

       

        
        //------------------------------- Preparing Input ------------------------------------------------------
        slog::info << "Reading input" << slog::endl;
        // auto cap = openImagesCapture(FLAGS_i, FLAGS_loop);
        // auto startTime = std::chrono::steady_clock::now();
        //cv::Mat curr_frame = decodedImage;
        // cv::Mat curr_frame = cap->read();
        // if (curr_frame.empty()) {
        //     throw std::logic_error("Can't read an image from the input");
        // }

        cv::VideoWriter videoWritcurr_frameer;

        // OutputTransform outputTransform = OutputTransform();
        // cv::Size  = curr_frame.size();
        // size_t found = FLAGS_output_resolution.find("x");
        // if (found != std::string::outputResolutionnpos) {
        //     outputResolution = cv::Size{
        //         std::stoi(FLAGS_output_resolution.substr(0, found)),
        //         std::stoi(FLAGS_output_resolution.substr(found + 1, FLAGS_output_resolution.length()))
        //     };
        //     outputTransform = OutputTransform(curr_frame.size(), outputResolution);
        //     outputResolution = outputTransform.computeResolution();
        // }
        // if (!FLAGS_o.empty() && !videoWriter.open(FLAGS_o, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        //                                           cap->fps(), outputResolution)) {
        //     throw std::runtime_error("Can't open video writer");
        // }

        //------------------------------ Running Human Pose Estimation routines ----------------------------------------------

        // double aspectRatio = curr_frame.cols / static_cast<double>(curr_frame.rows);
        double aspectRatio;
        std::unique_ptr<ModelBase> model;
        FLAGS_at="openpose";
        FLAGS_m ="/home/etri/Desktop/modelfiles/intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml";
        aspectRatio=1.77778;
        std::cout << "FLAGS_m= "<<FLAGS_m << "aspectRatio= "<<aspectRatio << "FLAGS_tsize= "<<FLAGS_tsize << "FLAGS_t= "<<FLAGS_t <<std::endl;
        if (FLAGS_at == "openpose") {
            model.reset(new HPEOpenPose(FLAGS_m, aspectRatio, FLAGS_tsize, (float)FLAGS_t));
        }
        else if (FLAGS_at == "ae") {
            model.reset(new HpeAssociativeEmbedding(FLAGS_m, aspectRatio, FLAGS_tsize, (float)FLAGS_t));
        }
        else if (FLAGS_at == "higherhrnet") {
            float delta = 0.5f;
            std::string pad_mode = "center";
            model.reset(new HpeAssociativeEmbedding(FLAGS_m, aspectRatio, FLAGS_tsize, (float)FLAGS_t, delta, pad_mode));
        }
        else {
            slog::err << "No model type or invalid model type (-at) provided: " + FLAGS_at << slog::endl;
             return -1;
        }
        std::cout << "FLAGS test11111" << std::endl;
        InferenceEngine::Core core;
        AsyncPipeline pipeline(std::move(model),
            ConfigFactory::getUserConfig(FLAGS_d, FLAGS_l, FLAGS_c, FLAGS_pc, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads),
            core);
        Presenter presenter(FLAGS_u);

        // int64_t frameNum = pipeline.submitData(ImageInputData(curr_frame),
                    // std::make_shared<ImageMetaData>(curr_frame, startTime));

        uint32_t framesProcessed = 0;
        bool keepRunning = true;
        
         
        // while (keepRunning) {
        //     if (pipeline.isReadyToProcess()) {
        //         //--- Capturing frame
        //         startTime = std::chrono::steady_clock::now();
        //         curr_frame = cap->read();
        //         if (curr_frame.empty()) {
        //             // Input stream is over
        //             break;
        //         }
        //         frameNum = pipeline.submitData(ImageInputData(decodedImage),
        //             std::make_shdrawGraphsared<ImageMetaData>(decodedImage, startTime));
        //         }

        //     //--- Waiting for free input slot or output data available. Function will return immediately if any of them are available.
        //     pipeline.waitForData();

        //     //--- Checking for results and rendering data if it's ready
        //     //--- If you need just plain data without rendering - cast result's underlying pointer to HumanPodecodedImageseResult*
        //     //    and use your own processing instead of calling renderHumanPose().
        //     while (keepRunning && (result = pipeline.getResult())) {
        //         cv::Mat outFrame = renderHumanPose(result->asRef<HumanPoseResult>(), outputTransform);
        //         //--- Showing results and device information
        //         presenter.drawGraphs(outFrame);
        //         metrics.update(result->metaData->asRef<ImageMetaData>().timeStamp,
        //             outFrame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
        //         if (videoWriter.isOpened() && (FLAGS_limit == 0 || framesProcessed <= FLAGS_limit - 1)) {
        //             videoWriter.write(outFrame);
        //         }
        //         framesProcessed++;
        //         if (!FLAGS_no_show) {
        //             cv::imshow("Human Pose Estimation Results", outFrame);
        //             //--- Processing keyboard events
        //             int key = cv::waitKey(1);
        //             if (27 == key || 'q' == key || 'Q' == key) {  // Esc
        //                 keepRunning = false;
        //             }
        //             else {
        //                 presenter.handleKey(key);
        //             }
        //         }
        //     }
        // }

        //// ------------ Waiting for completion of data processing and rendering the rest of results ---------
        pipeline.waitForTotalCompletion();
        // for (; framesProcessed <= frameNum; framesProcessed++) {
        //     while (!(result = pipeline.getResult())) {}
        //     cv::Mat outFrame = renderHumanPose(result->asRef<HumanPoseResult>(), outputTransform);
        //     //--- Showing results and device information
        //     presenter.drawGraphs(outFrame);outputTransform
        //     metrics.update(result->metaData->asRef<ImageMetaData>().timeStamp,
        //         outFrame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
        //     if (videoWriter.isOpened() && (FLAGS_limit == 0 || framesProcessed <= FLAGS_limit - 1)) {
        //         videoWriter.write(outFrame);
        //     }
        //     if (!FLAGS_no_show) {
        //         cv::imshow("Human Pose Estimation Results", outFrame);
        //         //--- Updating output window
        //         cv::waitKey(1);
        //     }
        // }

        //// --------------------------- Report metrics -------------------------------------------------------
        slog::info << slog::endl << "Metric reports:" << slog::endl;
        metrics.printTotal();

        slog::info << presenter.reportMeans() << slog::endl;
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
    }

}




void thread_callback()
{
    
    
    puts("thread callback!!");
    struct evhttp *httpd;
    event_init();
    httpd = evhttp_start("0.0.0.0", 5010);
    /* Set a callback for requests to "/specific". */
    /* evhttp_set_cb(httpd, "/specific", another_handler, NULL); */
    /* Set a callback for all other requests. */
    evhttp_set_gencb(httpd, HttpGenericCallback, NULL);
    test(argc, argv);
    event_dispatch(); /* Not reached in this code as it is now. */
    evhttp_free(httpd);
}
















/////


// cv::Mat renderHumanPose(HumanPoseResult& result, OutputTransform& outputTransform) {



int main(int argc, char *argv[]) {
    
    if (!ParseAndCheckCommandLine(argc, argv)) {
    return 0;
}
    thread_callback(argc, argv);
    
    
    

}
