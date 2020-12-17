// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// add comments

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <thread>

#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>

#include <ext_list.hpp>
#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "cnn.hpp"
#include "image_grabber.hpp"
#include "text_detection.hpp"
#include "text_recognition.hpp"

#include "text_detection_demo.hpp"
#include <evhttp.h>
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
}

using namespace InferenceEngine;

std::vector<cv::Point2f> floatPointsFromRotatedRect(const cv::RotatedRect &rect);
std::vector<cv::Point> boundedIntPointsFromRotatedRect(const cv::RotatedRect &rect, const cv::Size &image_size);
cv::Point topLeftPoint(const std::vector<cv::Point2f> &points, int *idx);
cv::Mat cropImage(const cv::Mat &image, const std::vector<cv::Point2f> &points, const cv::Size &target_size, int top_left_point_idx);
void setLabel(cv::Mat &im, const std::string label, const cv::Point &p);
cv::Mat picture_a;

Cnn text_detection, text_recognition;
std::map<std::string, InferencePlugin> plugins_for_devices;
std::vector<std::string> devices;

std::unique_ptr<Grabber> grabber;

float cls_conf_threshold;
float link_conf_threshold;
double text_detection_postproc_time = 0;
double text_recognition_postproc_time = 0;
double text_crop_time = 0;
double avg_time = 0;
const double avg_time_decay = 0.8;
const char kPadSymbol = '#';
std::string kAlphabet = std::string("0123456789abcdefghijklmnopqrstuvwxyz") + kPadSymbol;
double min_text_recognition_confidence = 0.0;
std::string Json_Rac[] ={"minX","minY","maxX","maxY"};

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

bool ParseAndCheckCommandLine(int argc, char *argv[])
{
    // ------------------------- Parsing and validating input arguments --------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h)
    {
        showUsage();
        return false;
    }

    //if (FLAGS_i.empty())
    //{
    //    throw std::logic_error("Parameter -i is not set");
    //}
    if (FLAGS_m_td.empty() && FLAGS_m_tr.empty())
    {
        throw std::logic_error("Neither parameter -m_td nor -m_tr is not set");
    }

    return true;
}

int clip(int x, int max_val)
{
    return std::min(std::max(x, 0), max_val);
}

std::string inference_fn(struct evbuffer *buf1)
{

    //auto image_path = FLAGS_i;
    //auto text_detection_model_path = FLAGS_m_td;
    //auto text_recognition_model_path = FLAGS_m_tr;
    //auto extension_path = FLAGS_l;
    //Cnn text_detection, text_recognition;
    // if (!FLAGS_m_td.empty())
    //     text_detection.Init(FLAGS_m_td, &plugins_for_devices[FLAGS_d_td], cv::Size(FLAGS_w_td, FLAGS_h_td));
    // if (!FLAGS_m_tr.empty())
    //     text_recognition.Init(FLAGS_m_tr, &plugins_for_devices[FLAGS_d_tr]);

    // if (!FLAGS_m_td.empty())
    //     text_detection.Init(FLAGS_m_td, &plugins_for_devices[FLAGS_d_td], cv::Size(FLAGS_w_td, FLAGS_h_td));
    // if (!FLAGS_m_tr.empty())
    //     text_recognition.Init(FLAGS_m_tr, &plugins_for_devices[FLAGS_d_tr]);
    //grabber = Grabber::make_grabber(FLAGS_dt, FLAGS_i);
    //grabber = Grabber::make_grabber(FLAGS_dt, FLAGS_i);
    std::string result;
    std::string buf;

    try
    {
        //grabber->GrabNextImage(&picture_a);

        if (!picture_a.empty())
        {
            cv::Mat demo_image = picture_a;
            cv::Size orig_image_size = picture_a.size();
            //std::chrono::steady_clock::time_point begin_frame = std::chrono::steady_clock::now();
            std::vector<cv::RotatedRect> rects;
            if (text_detection.is_initialized())
            {
                auto blobs = text_detection.Infer(picture_a);
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                rects = postProcess(blobs, orig_image_size, cls_conf_threshold, link_conf_threshold);
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                text_detection_postproc_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            }
            else
            {
                rects.emplace_back(cv::Point2f(0.0f, 0.0f), cv::Size2f(0.0f, 0.0f), 0.0f);
            }

            if (FLAGS_max_rect_num >= 0 && static_cast<int>(rects.size()) > FLAGS_max_rect_num)
            {
                std::sort(rects.begin(), rects.end(), [](const cv::RotatedRect &a, const cv::RotatedRect &b) {
                    return a.size.area() > b.size.area();
                });
                rects.resize(FLAGS_max_rect_num);
            }

            int num_found = text_recognition.is_initialized() ? 0 : static_cast<int>(rects.size());
            result = "{\"inferencetime\":"+std::to_string(text_recognition.time_elapsed() / text_recognition.ncalls()) +",\"data\": [";
            //result = "{\"inferencetime\":" + std::to_string(text_recognition.time_elapsed() / text_recognition.ncalls()) + "," + "\"data\": [";
            //result = "{\"inferencetime\":" + "," + "\"data\": [";

            for (const auto &rect : rects)
            {
                
                cv::Mat cropped_text;
                std::vector<cv::Point2f> points;
                int top_left_point_idx = -1;

                if (rect.size != cv::Size2f(0, 0) && text_detection.is_initialized())
                {
                    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                    points = floatPointsFromRotatedRect(rect);
                    topLeftPoint(points, &top_left_point_idx);
                    cropped_text = cropImage(picture_a, points, text_recognition.input_size(), top_left_point_idx);
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    text_crop_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                }
                else
                {
                    cropped_text = picture_a;
                }

                std::string res = "";
                double conf = 1.0;
                if (text_recognition.is_initialized())
                {
                    auto blobs = text_recognition.Infer(cropped_text);
                    auto output_shape = blobs.begin()->second->getTensorDesc().getDims();
                    if (output_shape[2] != kAlphabet.length())
                        throw std::runtime_error("The text recognition model does not correspond to alphabet.");

                    float *ouput_data_pointer = blobs.begin()->second->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
                    std::vector<float> output_data(ouput_data_pointer, ouput_data_pointer + output_shape[0] * output_shape[2]);

                    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                    res = CTCGreedyDecoder(output_data, kAlphabet, kPadSymbol, &conf);
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    text_recognition_postproc_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

                    res = conf >= min_text_recognition_confidence ? res : "";
                    num_found += !res.empty() ? 1 : 0;
                }

                if (FLAGS_r)
                {
                    

                    for (size_t i = 0; i < points.size(); i+=4)
                    {
                        if(i==0)
                        {
                             //result += +"{\"text\":"+ res+",";
                             result += +"{\"name\":\""+res+"\",";
                        }
                        //std::cout<<std::to_string(static_cast<int>(points[i].y))<<std::endl;
                        result += "\""+Json_Rac[i]+"\":"+std::to_string(static_cast<int>(points[i].x) ) 
                        +","+"\""+Json_Rac[i+1]+"\":"+ std::to_string(static_cast<int>(points[i+2].y))
                        +",\""+Json_Rac[i+2]+"\":"+std::to_string(static_cast<int>(points[i+2].x)) 
                        +","+"\""+Json_Rac[i+3]+"\":"+ std::to_string(static_cast<int>(points[i].y));
                       //result += "\"X"+std::to_string(i)+"\":"+std::to_string(clip(static_cast<int>(points[i].x), picture_a.cols - 1)) +",\"Y"+std::to_string(i)+"\":"+ std::to_string(clip(static_cast<int>(points[i].y), picture_a.rows - 1));
                        //std::cout << clip(static_cast<int>(points[i].x), picture_a.cols - 1) << "," << clip(static_cast<int>(points[i].y), picture_a.rows - 1);
                        if (i != points.size() - 4)
                        {
                            result +=",";     
                            //std::cout << ",";                           
                        }
                        else if(i == points.size() - 4)
                        {
                            result +="},";  
                        }

                       
                    }
                    buf = result.substr(0,result.length()-1);
                    buf += "]}";

                    //if (text_recognition.is_initialized())
                    //{
                    //    result += +",\"text\":"+ res;
                    //    //std::cout << "," << res;
                    //}

                  
                    //std::cout << "--------------------------------------------------------------------------" << std::endl;
                    
                }
                  
                //if (!FLAGS_no_show && (!res.empty() || !text_recognition.is_initialized()))
                //{
                //    for (size_t i = 0; i < points.size(); i++)
                //    {
                //        cv::line(demo_image, points[i], points[(i + 1) % points.size()], cv::Scalar(50, 205, 50), 2);
                //    }
//
                //    if (!points.empty() && !res.empty())
                //    {
                //        setLabel(demo_image, res, points[top_left_point_idx]);
                //    }
                //}
                
            }
            std::cout << buf << std::endl;
            //std::chrono::steady_clock::time_point end_frame = std::chrono::steady_clock::now();

            //if (avg_time == 0)
            //{
            //    avg_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end_frame - begin_frame).count());
            //}
            //else
            //{
            //    auto cur_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_frame - begin_frame).count();
            //    avg_time = avg_time * avg_time_decay + (1.0 - avg_time_decay) * cur_time;
            //}
            //int fps = static_cast<int>(1000 / avg_time);
           //if (!FLAGS_no_show)
           // {
           //     std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;
           //     //cv::putText(demo_image, "fps: " + std::to_string(fps) + " found: " + std::to_string(num_found),
           //     //          cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 1);
           //     cv::imshow("Press any key to exit", demo_image);
           //     char k = cv::waitKey(0);
           //     if (k == 27)
           //     {
           // 
           //     }
           // }

            //grabber->GrabNextImage(&picture_a);
        }

 //       if (text_detection.ncalls() && !FLAGS_r)
 //       {
 //           std::cout << "text detection model inference (ms) (fps): "
 //                     << text_detection.time_elapsed() / text_detection.ncalls() << " "
 //                     << text_detection.ncalls() * 1000 / text_detection.time_elapsed() << std::endl;
 //           if (std::fabs(text_detection_postproc_time) < std::numeric_limits<double>::epsilon())
 //           {
 //               throw std::logic_error("text_detection_postproc_time can't be equal to zero");
 //           }
 //           std::cout << "text detection postprocessing (ms) (fps): "
 //                     << text_detection_postproc_time / text_detection.ncalls() << " "
 //                     << text_detection.ncalls() * 1000 / text_detection_postproc_time << std::endl
 //                     << std::endl;
 //       }
//
 //       if (text_recognition.ncalls() && !FLAGS_r)
 //       {
 //           std::cout << "text recognition model inference (ms) (fps): "
 //                     << text_recognition.time_elapsed() / text_recognition.ncalls() << " "
 //                     << text_recognition.ncalls() * 1000 / text_recognition.time_elapsed() << std::endl;
 //           if (std::fabs(text_recognition_postproc_time) < std::numeric_limits<double>::epsilon())
 //           {
 //               throw std::logic_error("text_recognition_postproc_time can't be equal to zero");
 //           }
 //           std::cout << "text recognition postprocessing (ms) (fps): "
 //                     << text_recognition_postproc_time / text_recognition.ncalls() / 1000 << " "
 //                     << text_recognition.ncalls() * 1000000 / text_recognition_postproc_time << std::endl
 //                     << std::endl;
 //           if (std::fabs(text_crop_time) < std::numeric_limits<double>::epsilon())
 //           {
 //               throw std::logic_error("text_crop_time can't be equal to zero");
 //           }
 //           std::cout << "text crop (ms) (fps): " << text_crop_time / text_recognition.ncalls() / 1000 << " "
 //                     << text_recognition.ncalls() * 1000000 / text_crop_time << std::endl
 //                     << std::endl;
 //       }

        // ---------------------------------------------------------------------------------------------------
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        //return EXIT_FAILURE;
    }
    return buf;
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
             //for (int j = 0; j < nbytes; j++)
             //{
             //  printf("%c",cbuf[j]);
             //}

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
    printf("Odata: %d,%d,%d!!!!", nbytes, start_idx, end_idx);

    unsigned char img_bytes[img_byte_size];
    memset(img_bytes, 0, sizeof(unsigned char) * img_byte_size);
    memcpy(img_bytes, cbuf + (start_idx - 7), img_byte_size * sizeof(unsigned char));
    //FILE *tp_new = fopen("android_camera.jpg", "w");
    //if (tp_new)
    //{
    //    fwrite(img_bytes, 1, img_byte_size, tp_new);
    //    fclose(tp_new);
    //}

    cv::Mat rawData(1, img_byte_size, CV_8UC1, (void *)img_bytes);
    //printf("\n------------------%u------------------\n", rawData.data);
    cv::Mat decodedImage = imdecode(rawData, cv::IMREAD_COLOR);
    if (decodedImage.data == NULL)
    {
        puts("decodeddataNULL!!! \n");
    }

    static unsigned int count = 1;

    struct evbuffer *buf1 = evbuffer_new();
    picture_a = decodedImage;
    if (!buf1)
    {
        puts("failed to create response buffer \n");
        return;
    }
    else
    {
        std::string result = inference_fn(buf1);
        evbuffer_add_printf(buf1, "%s", result.c_str());
        //evhttp_send_reply(req, HTTP_OK, "OK", buf1);
        evhttp_send_reply(req, HTTP_OK, "OK", buf1);
        //evhttp_send_reply(req, 200, "OK", NULL);

        evbuffer_free(buf1);
    }

    printf("\n------------------%u------------------\n", count++);

   
}

void thread_callback()
{
    puts("thread callback!!");
    struct evhttp *httpd;
    event_init();
    httpd = evhttp_start("0.0.0.0", 5002);
    /* Set a callback for requests to "/specific". */
    /* evhttp_set_cb(httpd, "/specific", another_handler, NULL); */
    /* Set a callback for all other requests. */
    evhttp_set_gencb(httpd, HttpGenericCallback, NULL);

    event_dispatch(); /* Not reached in this code as it is now. */
    evhttp_free(httpd);
}

int main(int argc, char *argv[])
{
    try
    {
        // ----------------------------- Parsing and validating input arguments ------------------------------

        /** This demo covers one certain topology and cannot be generalized **/

        if (!ParseAndCheckCommandLine(argc, argv))
        {
            return 0;
        }

        // double text_detection_postproc_time = 0;
        // double text_recognition_postproc_time = 0;
        // double text_crop_time = 0;
        // double avg_time = 0;
        // const double avg_time_decay = 0.8;

        // const char kPadSymbol = '#';
        // std::string kAlphabet = std::string("0123456789abcdefghijklmnopqrstuvwxyz") + kPadSymbol;

        min_text_recognition_confidence = FLAGS_thr;

        //std::map<std::string, InferencePlugin> plugins_for_devices;
        devices = {FLAGS_d_td, FLAGS_d_tr};

        cls_conf_threshold = static_cast<float>(FLAGS_cls_pixel_thr);
        link_conf_threshold = static_cast<float>(FLAGS_link_pixel_thr);

        for (const auto &device : devices)
        {
            if (plugins_for_devices.find(device) != plugins_for_devices.end())
            {
                continue;
            }
            InferencePlugin plugin = PluginDispatcher().getPluginByDevice(device);
            /** Load extensions for the CPU plugin **/
            if ((device.find("CPU") != std::string::npos))
            {
                plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

                if (!FLAGS_l.empty())
                {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    plugin.AddExtension(extension_ptr);
                    slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
                }
            }
            else if (!FLAGS_c.empty())
            {
                // Load Extensions for other plugins not CPU
                plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            }
            plugins_for_devices[device] = plugin;
        }

        auto image_path = FLAGS_i;
        auto text_detection_model_path = FLAGS_m_td;
        auto text_recognition_model_path = FLAGS_m_tr;
        auto extension_path = FLAGS_l;
        //
        //Cnn text_detection, text_recognition;
        //std::cout<<plugins_for_devices[FLAGS_d_td]<<std::endl;
        if (!FLAGS_m_td.empty())
            text_detection.Init(FLAGS_m_td, &plugins_for_devices[FLAGS_d_td], cv::Size(FLAGS_w_td, FLAGS_h_td));
        if (!FLAGS_m_tr.empty())
            text_recognition.Init(FLAGS_m_tr, &plugins_for_devices[FLAGS_d_tr]);
        //

        // grabber = Grabber::make_grabber(FLAGS_dt, FLAGS_i);
        // std::unique_ptr<Grabber> grabber = Grabber::make_grabber(FLAGS_dt, FLAGS_i);

        //cv::Mat image;

        //     grabber->GrabNextImage(&image);

        //     while (!image.empty()) {
        //         cv::Mat demo_image = image.clone();
        //         cv::Size orig_image_size = image.size();

        //         std::chrono::steady_clock::time_point begin_frame = std::chrono::steady_clock::now();
        //         std::vector<cv::RotatedRect> rects;
        //         if (text_detection.is_initialized()) {
        //             auto blobs = text_detection.Infer(image);
        //             std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        //             rects = postProcess(blobs, orig_image_size, cls_conf_threshold, link_conf_threshold);
        //             std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //             text_detection_postproc_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        //         } else {
        //             rects.emplace_back(cv::Point2f(0.0f, 0.0f), cv::Size2f(0.0f, 0.0f), 0.0f);
        //         }

        //         if (FLAGS_max_rect_num >= 0 && static_cast<int>(rects.size()) > FLAGS_max_rect_num) {
        //             std::sort(rects.begin(), rects.end(), [](const cv::RotatedRect & a, const cv::RotatedRect & b) {
        //                 return a.size.area() > b.size.area();
        //             });
        //             rects.resize(FLAGS_max_rect_num);
        //         }

        //         int num_found = text_recognition.is_initialized() ? 0 : static_cast<int>(rects.size());

        //         for (const auto &rect : rects) {
        //             cv::Mat cropped_text;
        //             std::vector<cv::Point2f> points;
        //             int top_left_point_idx = -1;

        //             if (rect.size != cv::Size2f(0, 0) && text_detection.is_initialized()) {
        //                 std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        //                 points = floatPointsFromRotatedRect(rect);
        //                 topLeftPoint(points, &top_left_point_idx);
        //                 cropped_text = cropImage(image, points, text_recognition.input_size(), top_left_point_idx);
        //                 std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //                 text_crop_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        //             } else {
        //                 cropped_text = image;
        //             }

        //             std::string res = "";
        //             double conf = 1.0;
        //             if (text_recognition.is_initialized()) {
        //                 auto blobs = text_recognition.Infer(cropped_text);
        //                 auto output_shape = blobs.begin()->second->getTensorDesc().getDims();
        //                 if (output_shape[2] != kAlphabet.length())
        //                     throw std::runtime_error("The text recognition model does not correspond to alphabet.");

        //                 float *ouput_data_pointer = blobs.begin()->second->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
        //                 std::vector<float> output_data(ouput_data_pointer, ouput_data_pointer + output_shape[0] * output_shape[2]);

        //                 std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        //                 res = CTCGreedyDecoder(output_data, kAlphabet, kPadSymbol, &conf);
        //                 std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //                 text_recognition_postproc_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        //                 res = conf >= min_text_recognition_confidence ? res : "";
        //                 num_found += !res.empty() ? 1 : 0;
        //             }

        //             if (FLAGS_r) {
        //                 for (size_t i = 0; i < points.size(); i++) {
        //                     std::cout << clip(static_cast<int>(points[i].x), image.cols - 1) << "," <<
        //                                  clip(static_cast<int>(points[i].y), image.rows - 1);
        //                     if (i != points.size() - 1)
        //                         std::cout << ",";
        //                 }

        //                 if (text_recognition.is_initialized()) {
        //                     std::cout << "," << res;
        //                 }

        //                 std::cout << std::endl;
        //             }

        //             if (!FLAGS_no_show && (!res.empty() || !text_recognition.is_initialized())) {
        //                 for (size_t i = 0; i < points.size() ; i++) {
        //                     cv::line(demo_image, points[i], points[(i+1) % points.size()], cv::Scalar(50, 205, 50), 2);
        //                 }

        //                 if (!points.empty() && !res.empty()) {
        //                     setLabel(demo_image, res, points[top_left_point_idx]);
        //                 }
        //             }
        //         }

        //         std::chrono::steady_clock::time_point end_frame = std::chrono::steady_clock::now();

        //         if (avg_time == 0) {
        //             avg_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end_frame - begin_frame).count());
        //         } else {
        //             auto cur_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_frame - begin_frame).count();
        //             avg_time = avg_time * avg_time_decay + (1.0 - avg_time_decay) * cur_time;
        //         }
        //         //int fps = static_cast<int>(1000 / avg_time);

        //         if (!FLAGS_no_show) {
        //             std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;
        //             //cv::putText(demo_image, "fps: " + std::to_string(fps) + " found: " + std::to_string(num_found),
        //               //          cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 1);
        //             cv::imshow("Press any key to exit", demo_image);
        //             char k = cv::waitKey(0);
        //             if (k == 27) break;
        //         }

        //         grabber->GrabNextImage(&image);
        //     }

        //     if (text_detection.ncalls() && !FLAGS_r) {
        //       std::cout << "text detection model inference (ms) (fps): "
        //                 << text_detection.time_elapsed() / text_detection.ncalls() << " "
        //                 << text_detection.ncalls() * 1000 / text_detection.time_elapsed() << std::endl;
        //     if (std::fabs(text_detection_postproc_time) < std::numeric_limits<double>::epsilon()) {
        //         throw std::logic_error("text_detection_postproc_time can't be equal to zero");
        //     }
        //       std::cout << "text detection postprocessing (ms) (fps): "
        //                 << text_detection_postproc_time / text_detection.ncalls() << " "
        //                 << text_detection.ncalls() * 1000 / text_detection_postproc_time << std::endl << std::endl;
        //     }

        //     if (text_recognition.ncalls() && !FLAGS_r) {
        //       std::cout << "text recognition model inference (ms) (fps): "
        //                 << text_recognition.time_elapsed() / text_recognition.ncalls() << " "
        //                 << text_recognition.ncalls() * 1000 / text_recognition.time_elapsed() << std::endl;
        //       if (std::fabs(text_recognition_postproc_time) < std::numeric_limits<double>::epsilon()) {
        //           throw std::logic_error("text_recognition_postproc_time can't be equal to zero");
        //       }
        //       std::cout << "text recognition postprocessing (ms) (fps): "
        //                 << text_recognition_postproc_time / text_recognition.ncalls() / 1000 << " "
        //                 << text_recognition.ncalls() * 1000000 / text_recognition_postproc_time << std::endl << std::endl;
        //       if (std::fabs(text_crop_time) < std::numeric_limits<double>::epsilon()) {
        //           throw std::logic_error("text_crop_time can't be equal to zero");
        //       }
        //       std::cout << "text crop (ms) (fps): " << text_crop_time / text_recognition.ncalls() / 1000 << " "
        //                 << text_recognition.ncalls() * 1000000 / text_crop_time << std::endl << std::endl;
        //     }

        //     // ---------------------------------------------------------------------------------------------------
        std::thread _t1(thread_callback);
        _t1.join();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "error!!" << std::endl;
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

std::vector<cv::Point2f> floatPointsFromRotatedRect(const cv::RotatedRect &rect)
{
    cv::Point2f vertices[4];
    rect.points(vertices);

    std::vector<cv::Point2f> points;
    for (int i = 0; i < 4; i++)
    {
        points.emplace_back(vertices[i].x, vertices[i].y);
    }
    return points;
}

cv::Point topLeftPoint(const std::vector<cv::Point2f> &points, int *idx)
{
    cv::Point2f most_left(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    cv::Point2f almost_most_left(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

    int most_left_idx = -1;
    int almost_most_left_idx = -1;

    for (size_t i = 0; i < points.size(); i++)
    {
        if (most_left.x > points[i].x)
        {
            if (most_left.x != std::numeric_limits<float>::max())
            {
                almost_most_left = most_left;
                almost_most_left_idx = most_left_idx;
            }
            most_left = points[i];
            most_left_idx = i;
        }
        if (almost_most_left.x > points[i].x && points[i] != most_left)
        {
            almost_most_left = points[i];
            almost_most_left_idx = i;
        }
    }

    if (almost_most_left.y < most_left.y)
    {
        most_left = almost_most_left;
        most_left_idx = almost_most_left_idx;
    }

    *idx = most_left_idx;
    return most_left;
}

cv::Mat cropImage(const cv::Mat &image, const std::vector<cv::Point2f> &points, const cv::Size &target_size, int top_left_point_idx)
{
    cv::Point2f point0 = points[top_left_point_idx];
    cv::Point2f point1 = points[(top_left_point_idx + 1) % 4];
    cv::Point2f point2 = points[(top_left_point_idx + 2) % 4];

    cv::Mat crop(target_size, CV_8UC3, cv::Scalar(0));

    std::vector<cv::Point2f> from{point0, point1, point2};
    std::vector<cv::Point2f> to{cv::Point2f(0.0f, 0.0f), cv::Point2f(static_cast<float>(target_size.width - 1), 0.0f),
                                cv::Point2f(static_cast<float>(target_size.width - 1), static_cast<float>(target_size.height - 1))};

    cv::Mat M = cv::getAffineTransform(from, to);

    cv::warpAffine(image, crop, M, crop.size());

    return crop;
}

void setLabel(cv::Mat &im, const std::string label, const cv::Point &p)
{
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.7;
    int thickness = 1;
    int baseline = 0;

    cv::Size text_size = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    auto text_position = p;
    text_position.x = std::max(0, p.x);
    text_position.y = std::max(text_size.height, p.y);

    cv::rectangle(im, text_position + cv::Point(0, baseline), text_position + cv::Point(text_size.width, -text_size.height), CV_RGB(50, 205, 50), cv::FILLED);
    cv::putText(im, label, text_position, fontface, scale, CV_RGB(255, 255, 255), thickness, 8);
}
