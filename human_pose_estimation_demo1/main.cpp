// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//
/**
* \brief The entry point for the Inference Engine Human Pose Estimation demo application
* \file human_pose_estimation_demo/main.cpp
* \example human_pose_estimation_demo/main.cpp
*/

#include <vector>
#include <time.h>
#include <thread>
#include <inference_engine.hpp>
#include <samples/ocv_common.hpp>
#include "human_pose_estimation_demo.hpp"
#include "human_pose_estimator.hpp"
#include "render_human_pose.hpp"
#include <evhttp.h>

//support c language
extern "C"
{
#include <sys/types.h>
#include <sys/time.h>
#include <sys/queue.h>
#include <stdlib.h>
#include <err.h>
#include <event.h>
#include <evhttp.h>
#include <cstdlib>
#include <signal.h>
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

using namespace InferenceEngine;
using namespace human_pose_estimation;
//

cv::Mat decodedImage;
int delay = 33;
double inferenceTime = 0.0;
std::stringstream rawPose;
int break_point=0;

//HumanPoseEstimator estimator("\0","\0",false);
void generic_cb(struct evhttp_request *req, void *arg)
{

    /* Response */
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


bool ParseAndCheckCommandLine(int argc, char *argv[])
{
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h)
    {
        showUsage();
        return false;
    }

    std::cout << "[ INFO ] Parsing input parameters" << std::endl;

    if (FLAGS_i.empty())
    {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty())
    {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

void HttpGenericCallback(struct evhttp_request *req, void *arg)
{
    const char *cmdtype;
    struct evkeyvalq *headers;
    struct evkeyval *header;
    struct evbuffer *buf;

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

    static unsigned int count = 1;

    struct evbuffer *buf1 = evbuffer_new();
    //picture_a = decodedImage;
    if (!buf1)
    {
        puts("failed to create response buffer \n");
        return;
    }
    else
    {
     
        std::cout << rawPose.str().c_str() << std::endl;
        rawPose.str().c_str();
        evbuffer_add_printf(buf1, "%s", rawPose.str().c_str());
        //evhttp_send_reply(req, HTTP_OK, "OK", buf1);
        evhttp_send_reply(req, HTTP_OK, "OK", buf1);
        //evhttp_send_reply(req, 200, "OK", NULL);
        rawPose.str("");
        rawPose.clear();
        evbuffer_free(buf1);
    }

    printf("\n------------------%u------------------\n", count++);
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

    event_dispatch(); /* Not reached in this code as it is now. */
    evhttp_free(httpd);
}

void thread_test(HumanPoseEstimator estimator)
{
    while (1)
    {
        //std::cout << "thread start!!!" << std::endl;
        if (!decodedImage.empty())
        {
            try
            {
                double t1 = static_cast<double>(cv::getTickCount());
                std::vector<HumanPose> poses = estimator.estimate(decodedImage);
                double t2 = static_cast<double>(cv::getTickCount());
                if (inferenceTime == 0)
                {
                    inferenceTime = (t2 - t1) / cv::getTickFrequency() * 1000;
                }
                else
                {
                    inferenceTime = inferenceTime * 0.95 + 0.05 * (t2 - t1) / cv::getTickFrequency() * 1000;
                }
                    rawPose << "{\"inferencetime\":" << inferenceTime << ","
                            << "\"data\":[";
                    unsigned int id_pose = 0;
                    for (HumanPose const &pose : poses)
                    {
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
                        if (id_pose != poses.size())
                            rawPose << ",";
                    }
                    rawPose << "]}";
                
               
            }
            catch (const std::exception &error)
            {
                std::cerr << "[ ERROR ] " << error.what() << std::endl;
                //return EXIT_FAILURE;
            }
            catch (...)
            {
                std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
                //return EXIT_FAILURE;
            }
        }
        else
        {
            //std::cout << "image.empty!" << std::endl;
        }
        sleep(0.01);
    }
}

int main(int argc, char *argv[])
{
    try
    {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;
        //HumanPoseEstimator estimator(FLAGS_m, FLAGS_d, FLAGS_pc);

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv))
        {
            return EXIT_SUCCESS;
        }
       
        HumanPoseEstimator estimator(FLAGS_m, FLAGS_d, FLAGS_pc);
        
        setsid();
        umask(0);

        breakCapture = signal(SIGINT, signalingHandler);
        std::thread _t2(thread_callback);
        std::thread _t1(thread_test, estimator);

        _t2.join();
        _t1.join();
        std::cout << "[ INFO ] Execution successful" << std::endl;
        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }
}
