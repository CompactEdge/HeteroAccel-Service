// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine object_detection demo application
* \file object_detection_demo_ssd_async/main.cpp
* \example object_detection_demo_ssd_async/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <thread>
#include <time.h>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
#include <chrono>
#include <vector>

#include "object_detection_demo_ssd_async.hpp"
#include <ext_list.hpp>

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

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
cv::Mat picture_a;
double ocv_render_time = 0;
CNNNetReader netReader;
InferRequest::Ptr async_infer_request_curr;
std::stringstream rawFace;

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

bool ParseAndCheckCommandLine(int argc, char *argv[])
{
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h)
    {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    //if (FLAGS_i.empty())
    //{
    //    throw std::logic_error("Parameter -i is not set");
    //}

    if (FLAGS_m.empty())
    {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

void frameToBlob(const cv::Mat &frame,
                 InferRequest::Ptr &inferRequest,
                 const std::string &inputName)
{
    if (FLAGS_auto_resize)
    {
        /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
        inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
    }
    else
    {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, frameBlob);
    }
}

std::string inference_fn(struct evbuffer *buf1)
{
    
    const size_t width = 256;
    const size_t height = 256;
    InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
    // InputInfo::Ptr &input = inputInfo.begin()->second;
    auto inputName = inputInfo.begin()->first;
    OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
    DataPtr &output = outputInfo.begin()->second;
    auto outputName = outputInfo.begin()->first;
    // const int num_classes = netReader.getNetwork().getLayerByName(outputName.c_str())->GetParamAsInt("num_classes");
    const SizeVector outputDims = output->getTensorDesc().getDims();
    const int maxProposalCount = outputDims[2];
    const int objectSize = outputDims[3];

    auto t0 = std::chrono::high_resolution_clock::now();
    frameToBlob(picture_a, async_infer_request_curr, inputName);

    // Main sync point:
    // in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
    // in the regular mode we start the CURRENT request and immediately wait for it's completion
    std::string result; 
    std::string buf;
    async_infer_request_curr->StartAsync();
    if (OK == async_infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY))
    {
        

        // t0 = std::chrono::high_resolution_clock::now();
        // ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
        // wallclock = t0;
        
        
        //ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();
        
        //ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();

        
        
         //std::ostringstream out;
         //out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)<< (ocv_decode_time + ocv_render_time) << " ms";
        // cv::putText(curr_frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0));
        // out.str("");
        // out << "Wallclock time " << (isAsyncMode ? "(TRUE ASYNC):      " : "(SYNC, press Tab): ");
        // out << std::fixed << std::setprecision(2) << wall.count() << " ms (" << 1000.f / wall.count() << " fps)";
        // cv::putText(curr_frame, out.str(), cv::Point2f(0, 50), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255));
        // if (!isAsyncMode)
        // { // In the true async mode, there is no way to measure detection time directly
        //     out.str("");
        //     out << "Detection time  : " << std::fixed << std::setprecision(2) << detection.count()
        //         << " ms ("
        //         << 1000.f / detection.count() << " fps)";
        //     cv::putText(curr_frame, out.str(), cv::Point2f(0, 75), cv::FONT_HERSHEY_TRIPLEX, 0.6,
        //                 cv::Scalar(255, 0, 0));
        // }

        // ---------------------------Process output blobs--------------------------------------------------
        // Processing results of the CURRENT request
        const float *detections = async_infer_request_curr->GetBlob(outputName)->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();


        auto t1 = std::chrono::high_resolution_clock::now();
        ms detectiontime = std::chrono::duration_cast<ms>(t1 - t0);

        result = "{\"inferencetime\":" + std::to_string(detectiontime.count()) +  ","
                                + "\"data\":[ ";
        for (int i = 0; i < maxProposalCount; i++)
        {
            float image_id = detections[i * objectSize + 0];
            if (image_id < 0)
            {
                break;
            }

            float confidence = detections[i * objectSize + 2];
            //auto label = static_cast<int>(detections[i * objectSize + 1]);
            float xmin = detections[i * objectSize + 3] * width;
            float ymin = detections[i * objectSize + 4] * height;
            float xmax = detections[i * objectSize + 5] * width;
            float ymax = detections[i * objectSize + 6] * height;


            //t1 = std::chrono::high_resolution_clock::now();
            //ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();
            //if (FLAGS_r)
            //{
            //    std::cout << "[" << i << "," << label << "] element, prob = " << confidence << "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")"
            //              << ((confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
            //}

            if (confidence > FLAGS_t)
            {
                result += "{\"score\":"+ std::to_string(confidence) + ","
                        + "\"minX\": " + std::to_string(xmin) + ","
                        + "\"minY\": " + std::to_string(ymin) + ","
                        + "\"maxX\": " + std::to_string(xmax) + ","
                        + "\"maxY\": " + std::to_string(ymax)
                        + "},";
                // std::cout << "[xmin,ymin,xmax,ymax]=" <<confidence<< xmin << ", " << ymin << ", " << xmax << ", " << ymax <<", "<< ocv_render_time <<std::endl;
            
        if(i%30==0)
         {
            std::cout << result<< std::endl;
         }
            }
        }
         //rawFace = rawFace.str().substr(0,rawFace.str().length()-1);
         buf = result.substr(0,result.length()-1);
         buf += "]}";
  
         
        
    }
    return buf;
    //cv::imshow("Detection results", curr_frame);

    // t1 = std::chrono::high_resolution_clock::now();
    // ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();
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
    printf("Odata: %d,%d,%d", nbytes, start_idx, end_idx);

    unsigned char img_bytes[img_byte_size];
    memset(img_bytes, 0, sizeof(unsigned char) * img_byte_size);
    memcpy(img_bytes, cbuf + (start_idx - 7), img_byte_size * sizeof(unsigned char));
    FILE *tp_new = fopen("android_camera.png", "w");
    if (tp_new)
    {
        fwrite(img_bytes, 1, img_byte_size, tp_new);
        fclose(tp_new);
    }

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
        evbuffer_add_printf(buf1, "%s",result.c_str() );
        //evhttp_send_reply(req, HTTP_OK, "OK", buf1);
        evhttp_send_reply(req, HTTP_OK, "OK", buf1);
        //evhttp_send_reply(req, 200, "OK", NULL);

        evbuffer_free(buf1);
    }

    printf("\n------------------%u------------------\n", count++);

    //RunMPPGraph(&decodedImage);
    
}

void thread_callback()
{
    puts("thread callback!!");
    struct evhttp *httpd;
    event_init();
    httpd = evhttp_start("0.0.0.0", 5004);
    /* Set a callback for requests to "/specific". */
    /* evhttp_set_cb(httpd, "/specific", another_handler, NULL); */
    /* Set a callback for all other requests. */
    evhttp_set_gencb(httpd, HttpGenericCallback, NULL);

    event_dispatch(); /* Not reached in this code as it is now. */
    evhttp_free(httpd);
}



int main(int argc, char *argv[])
{

    /** This demo covers certain topology and cannot be generalized for any object detection **/
    std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

    // ------------------------------ Parsing and validation of input args ---------------------------------
    if (!ParseAndCheckCommandLine(argc, argv))
    {
        //return 0;
    }

    slog::info << "Reading input" << slog::endl;
    //cv::VideoCapture cap;
    //if (!((FLAGS_i == "cam") ? cap.open(0) : cap.open(FLAGS_i.c_str()))) {
    //    throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
    //}
    //const size_t width = (size_t)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    //const size_t height = (size_t)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    // const size_t width = 480;
    // const size_t height = 480;

    // --------------------------- 1. Load Plugin for inference engine -------------------------------------
    slog::info << "Loading plugin" << slog::endl;
    InferencePlugin plugin = PluginDispatcher().getPluginByDevice(FLAGS_d);
    printPluginVersion(plugin, std::cout);

    /** Load extensions for the plugin **/

    /** Loading default extensions **/
    if (FLAGS_d.find("CPU") != std::string::npos)
    {
        /**
         * cpu_extensions library is compiled from "extension" folder containing
         * custom MKLDNNPlugin layer implementations. These layers are not supported
         * by mkldnn, but they can be useful for inferring custom topologies.
        **/
        plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
    }

    if (!FLAGS_l.empty())
    {
        // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
        IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l.c_str());
        plugin.AddExtension(extension_ptr);
    }
    if (!FLAGS_c.empty())
    {
        // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
        plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
    }

    /** Per layer metrics **/
    if (FLAGS_pc)
    {
        plugin.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
    slog::info << "Loading network files" << slog::endl;

    /** Read network model **/
    netReader.ReadNetwork(FLAGS_m);
    /** Set batch size to 1 **/
    slog::info << "Batch size is forced to  1." << slog::endl;
    netReader.getNetwork().setBatchSize(1);
    /** Extract model name and load it's weights **/
    std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
    netReader.ReadWeights(binFileName);
    /** Read labels (if any)**/
    std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
    std::vector<std::string> labels;
    std::ifstream inputFile(labelFileName);
    std::copy(std::istream_iterator<std::string>(inputFile),
              std::istream_iterator<std::string>(),
              std::back_inserter(labels));
    // -----------------------------------------------------------------------------------------------------

    /** SSD-based network should have one input and one output **/
    // --------------------------- 3. Configure input & output ---------------------------------------------
    // --------------------------- Prepare input blobs -----------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
    if (inputInfo.size() != 1)
    {
        throw std::logic_error("This demo accepts networks having only one input");
    }
    InputInfo::Ptr &input = inputInfo.begin()->second;
    auto inputName = inputInfo.begin()->first;
    input->setPrecision(Precision::U8);
    if (FLAGS_auto_resize)
    {
        input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        input->getInputData()->setLayout(Layout::NHWC);
    }
    else
    {
        input->getInputData()->setLayout(Layout::NCHW);
    }
    // --------------------------- Prepare output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
    if (outputInfo.size() != 1)
    {
        throw std::logic_error("This demo accepts networks having only one output");
    }
    DataPtr &output = outputInfo.begin()->second;
    auto outputName = outputInfo.begin()->first;
    const int num_classes = netReader.getNetwork().getLayerByName(outputName.c_str())->GetParamAsInt("num_classes");
    if (static_cast<int>(labels.size()) != num_classes)
    {
        if (static_cast<int>(labels.size()) == (num_classes - 1)) // if network assumes default "background" class, having no label
            labels.insert(labels.begin(), "fake");
        else
            labels.clear();
    }
    const SizeVector outputDims = output->getTensorDesc().getDims();
    // const int maxProposalCount = outputDims[2];
    const int objectSize = outputDims[3];
    if (objectSize != 7)
    {
        throw std::logic_error("Output should have 7 as a last dimension");
    }
    if (outputDims.size() != 4)
    {
        throw std::logic_error("Incorrect output dimensions for SSD");
    }
    output->setPrecision(Precision::FP32);
    output->setLayout(Layout::NCHW);
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 4. Loading model to the plugin ------------------------------------------
    slog::info << "Loading model to the plugin" << slog::endl;
    ExecutableNetwork network = plugin.LoadNetwork(netReader.getNetwork(), {});
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 5. Create infer request -------------------------------------------------
    async_infer_request_curr = network.CreateInferRequestPtr();
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 6. Do inference ---------------------------------------------------------
    slog::info << "Start inference " << slog::endl;

    // auto total_t0 = std::chrono::high_resolution_clock::now();
    // auto wallclock = std::chrono::high_resolution_clock::now();
    // double ocv_decode_time = 0, ocv_render_time = 0;

    std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;

    // std::thread _t1(objectdetection_th, argc, argv);
    std::thread _t2(thread_callback);

    // _t1.join();
    _t2.join();

    // objectdetection_th(argc,argv);
    //slog::info << "Execution successful" << slog::endl;
    //return 0;
}
