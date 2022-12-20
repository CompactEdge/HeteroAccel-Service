// Copyright 2019 The MediaPipe Authors.
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
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
// This example requires a linux computer and a GPU with EGL support drivers.
#include <cstdlib>
#include <evhttp.h> // add ed evhttp.h 
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
//#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>
#include <thread>

//#include "mediapipe/framework/b64.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include <err.h>

#include <evhttp.h>
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/framework/formats/landmark.pb.h"

#define SVR_IP "127.0.0.1"
#define SVR_PORT 8080
char *test;
constexpr char kInputStream[] = "input_video";
// constexpr char kOutputStream[] = "output_video";
constexpr char kMultiHandLandmarksOutputStream[] = "multi_hand_landmarks";
constexpr char kWindowName[] = "MediaPipe";
cv::Mat picture_a;
double inferenceTime = 0.0;
#define BUFFER_SIZE 4096
DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
DEFINE_string(output_video_path, "",
              "Full path of where to save result (.mp4 only). "
              "If not provided, show result in a window.");

void specific_cb(struct evhttp_request *req, void *arg)
{
  struct evbuffer *evbuf;

  if ((evbuf = evbuffer_new()) == NULL)
  {
    printf("evbuffer_new() failed");
    evhttp_send_reply(req, HTTP_INTERNAL, "Internal error", NULL);
    return;
  }

  /* Body */
  evbuffer_add_printf(evbuf, "It's work!");

  /* Response */
  evhttp_send_reply(req, HTTP_OK, "OK", evbuf);

  /* Free resource */
  evbuffer_free(evbuf);
}

void generic_cb(struct evhttp_request *req, void *arg)
{
// media pipe source
  /* Response */
  evhttp_send_reply(req, HTTP_NOTFOUND, "Not found", NULL);
}

::mediapipe::Status RunMPPGraph() //cv::Mat *imgMat)
{

  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));

  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  puts("Get calculator graph config contents:");
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  puts("Initialize the calculator graph.");
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the GPU.";
  puts("Initialize the GPU");
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  mediapipe::GlCalculatorHelper gpu_helper;
  gpu_helper.InitializeForTest(graph.GetGpuResources().get());

  // LOG(INFO) << "Initialize the camera or load the video.";
  // cv::VideoCapture capture;
  // const bool load_video = !FLAGS_input_video_path.empty();

  //if (load_video)
  //{
  //  capture.open(FLAGS_input_video_path);
  //}
  //else
  //{
  //  LOG(INFO) << "FIND VIDEO CAM!!!.";
  //  capture.open(0);
  //}
  //RET_CHECK(capture.isOpened());
  // cv::VideoWriter writer;
  //   const bool save_video = !FLAGS_output_video_path.empty();
  //   if (!save_video) {
  //     cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
  // #if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
  //     capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  //     capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  //     capture.set(cv::CAP_PROP_FPS, 30);
  // #endif
  //   }

  LOG(INFO) << "Start running the calculator graph.";
  puts("Start running the calculator graph.");
  // ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller, graph.AddOutputStreamPoller(kOutputStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller multi_hand_landmarks_poller, graph.AddOutputStreamPoller(kMultiHandLandmarksOutputStream));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  puts("Start grabbing and processing frames.");

  bool grab_frames = true;
  while (grab_frames)
  {

    // Capture opencv camera or video frame.
    //cv::Mat camera_frame_raw;
    // capture >> camera_frame_raw; // get images
    if (picture_a.empty())
    {
      //break; // End of video.
    }
    else
    {
      //cv::imwrite("imgMat.png",picture_a);
      cv::Mat camera_frame;
      //
      cv::cvtColor(picture_a, camera_frame, cv::COLOR_BGR2RGB); // BGR이면 필요 RGB면 지워도됨
      //if (!load_video)
      //{
      //  cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
      //}

      // Wrap Mat into an ImageFrame.s
      auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
          mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
          mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
      cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
      camera_frame.copyTo(input_frame_mat);
      double t1 = static_cast<double>(cv::getTickCount());
      // Prepare and add graph input packet.
      size_t frame_timestamp_us =
          (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
      MP_RETURN_IF_ERROR(
          gpu_helper.RunInGlContext([&input_frame, &frame_timestamp_us, &graph,
                                     &gpu_helper]() -> ::mediapipe::Status {
            // Convert ImageFrame to GpuBuffer.
            auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
            auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
            glFlush();
            texture.Release();
            // Send GPU image packet into the graph.
            MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
                kInputStream, mediapipe::Adopt(gpu_frame.release())
                                  .At(mediapipe::Timestamp(frame_timestamp_us))));
            return ::mediapipe::OkStatus();
          }));
      double t2 = static_cast<double>(cv::getTickCount());
      if (inferenceTime == 0)
      {
        inferenceTime = (t2 - t1) / cv::getTickFrequency() * 1000;
      }
      else
      {
        inferenceTime = inferenceTime * 0.95 + 0.05 * (t2 - t1) / cv::getTickFrequency() * 1000;
      }

      ::mediapipe::Packet multi_hand_landmarks_packet;
      if (!multi_hand_landmarks_poller.Next(&multi_hand_landmarks_packet))
      {
        puts("landmarks error!!!!!!");
        //break;
      }

      const auto &multi_hand_landmarks = multi_hand_landmarks_packet.Get<std::vector<::mediapipe::NormalizedLandmarkList>>();
      if (multi_hand_landmarks.size())
      {
        char key_x[BUFFER_SIZE] = "\0";
        char key_y[BUFFER_SIZE] = "\0";
        char InferenceTime[BUFFER_SIZE] = "\0";
        char dataName[BUFFER_SIZE] = "\0";
        char *json_content = (char *)malloc(BUFFER_SIZE * multi_hand_landmarks.size() * sizeof(char *));
        //strcpy(&json_content[0], "{");
        //strcpy(&json_content[0], "[");
        //strcpy(&json_content[0], "{\"inferencetime\":" << inferenceTime <<  ","
        //                          << "\"data\":[");
        strcpy(&json_content[0], "{\"inferencetime\":");
        sprintf(InferenceTime, "%f,", inferenceTime);
        strcpy(dataName, "\"data\":[");
        strcat(json_content, InferenceTime);
        strcat(json_content, dataName);
        int hand_id = 0;
        int hand_count = 0;
        for (const auto &single_hand_landmarks : multi_hand_landmarks)
        {

          //strcpy(key_x, "[{'x':[");
          //strcpy(key_y, "'y':[");
          strcpy(key_x, "{\"x\":[");
          strcpy(key_y, "\"y\":[");
          //LOG(INFO) << "Hands" << multi_hand_landmarks.size() << "]:";
          // LOG(INFO) << "Hand index = [" << hand_id << "]:";
          // std::cout << single_hand_landmarks.landmark_size() << std::endl;
          for (int i = 0; i < single_hand_landmarks.landmark_size(); ++i)
          {
            const auto &landmark = single_hand_landmarks.landmark(i);
            //LOG(INFO) << "\tLandmark [" << i << "]: (" << landmark.x() << ", " << landmark.y() << ", " << landmark.z() << ")";
            char x_str[100] = "\0";
            char y_str[100] = "\0";
            if (i == single_hand_landmarks.landmark_size() - 1)
            {

              sprintf(x_str, "%f],", landmark.x());
              //sprintf(y_str, "%f]}]", landmark.y());
              if (multi_hand_landmarks.size() == 1)
              {
                sprintf(y_str, "%f]}", landmark.y());
              }
              else if (multi_hand_landmarks.size() == 2 && hand_id == 1)
              {
                sprintf(y_str, "%f]}", landmark.y());
              }
              else
              {
                sprintf(y_str, "%f]},", landmark.y());
              }
            }
            else
            {
              sprintf(x_str, "%f,", landmark.x());
              sprintf(y_str, "%f,", landmark.y());
            }
            strcat(key_x, x_str);
            strcat(key_y, y_str);
          }
          strcat(json_content, key_x);
          strcat(json_content, key_y);
          ++hand_id;

          memset(key_x, 0, BUFFER_SIZE * sizeof(char));
          memset(key_y, 0, BUFFER_SIZE * sizeof(char));
        }
        strcat(json_content, "]}");
        //strcat(json_content, "}");
        //std::cout << json_content << std::endl;
        test = json_content;
        std::cout << test << std::endl;
        //    struct event_base *evbase;
        //    struct evhttp     *evhttp;
        //
        //    /* Init event base */
        //    if ((evbase = event_base_new()) == NULL) {
        //        printf("event_base_new() failed\n");
        //
        //    }
        //  printf("66666666666666666666666666666");
        //    /* Init evhttp */
        //    if ((evhttp = evhttp_new(evbase)) == NULL) {
        //        printf("evhttp_new() failed\n");
        //
        //    }
        //    /* Set server IP, port */
        //    if (evhttp_bind_socket(evhttp, SVR_IP, SVR_PORT) == -1) {
        //        printf("evhttp_bind_socket() failed\n");
        //
        //    } else {
        //        printf("Listening on [%s:%d]\n", SVR_IP, SVR_PORT);
        //    }
        //      printf("77777777777777777777777");
        //    /* Set a callback for specific path */
        //    if (evhttp_set_cb(evhttp, "/test", specific_cb, NULL) < 0) {
        //        printf("evhttp_set_cb() failed\n");
        //    }
        //     printf("888888888888888888");
        //    /* Set a callback for default path */
        //    evhttp_set_gencb(evhttp, generic_cb, NULL);
        //     printf("9999999999999999999999");
        //    /* Enter event loop */
        //    event_base_dispatch(evbase);
        //     printf("10101010010101011010");
        //    /* Free resource */
        //    evhttp_free(evhttp);
        //         printf("aaaaaaaaaaaaa");
        //    event_base_free(evbase);
        //
        //
        //
        //

        //send json_content to server via libevent
        //send_json(json_content);
        free(json_content); //send
      }
    }
    sleep(0.01);
  }

  LOG(INFO) << "Shutting down.";
  puts("Shutting down");
  // if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  //return graph.WaitUntilDone();
  //////////////////////////////////////////////////////////
}

//find string
int findString(char *str, char *sub)
{
  char *p1, *p2, *p3;
  int i = 0, j = 0, flag = 0;

  p1 = str;
  p2 = sub;

  for (i = 0; i < strlen(str); i++)
  {
    if (*p1 == *p2)
    {
      p3 = p1;
      for (j = 0; j < strlen(sub); j++)
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
      if (j == strlen(sub))
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
  int remaind = 0;
  char *sub_start = "stream";
  char *sub_end = "--";
  //char *sub_end = "ho";
  char cbuf[evbuffer_get_length(buf)];
  memset(cbuf, 0, sizeof(char) * evbuffer_get_length(buf));
  int start_idx = 0;
  int end_idx = 0;
  int nbytes;

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
  // printf("Odata: %d,%d,%d",nbytes,start_idx,end_idx);

  unsigned char img_bytes[img_byte_size];
  memset(img_bytes, 0, sizeof(unsigned char) * img_byte_size);
  memcpy(img_bytes, cbuf + (start_idx - 7), img_byte_size * sizeof(unsigned char));
  // FILE *tp_new = fopen("android_camera.png","w");
  // if (tp_new)
  // {
  //     fwrite(img_bytes , 1, img_byte_size, tp_new);
  //     fclose(tp_new);
  // }
  // FILE *tp = fopen("img1.png","wb");
  // if (tp)
  // {
  //     fwrite(cbuf , 1, nbytes, tp);
  //     fclose(tp);
  // }

  //cv::Mat in_img = cv::imread("android_camera.png");
  // cv::Mat img3(256, 256, CV_8SC2);
  // std::cout<<"channels:"<<img3.channels()<<std::endl;
  //std::cout<<"bytes:"<<img_byte_size<<std::endl;
  //cv::Mat rawData(1,img_byte_size,CV_8SC1,(void*)img_bytes);
  // cv::Mat rawData(256,256,CV_8UC3,(unsigned char*)img_bytes);
  //std::cout<<"channels:"<<rawData.channels()<<std::endl;
  // img3= cv::imdecode(rawData,cv::IMREAD_UNCHANGED);

  //cv::Mat my_mat (256, 256, CV_8SC1, &img_bytes[0]); // BGR 이미지의 경우 CV_8UC3 사용
  //cv::Mat my_mat (256, 256, CV_8SC3,&img_bytes); // BGR 이미지의 경우 CV_8UC3 사용
  //cv::Mat decodedMat= imdecode(rawData,cv::IMREAD_ANYDEPTH);
  //printf("decodeMatdata=%d\n",sizeof(img_byte_size));
  //cv::imshow("input",rawData);
  //cv::imshow("awge",in_img);
  //cv::waitKey(0);
  //cv::imwrite("img.png",decodedMat);
  cv::Mat rawData(1, img_byte_size, CV_8UC1, (void *)img_bytes);
  //printf("\n------------------%u------------------\n", rawData.data);
  cv::Mat decodedImage = imdecode(rawData, cv::IMREAD_COLOR);
  if (decodedImage.data == NULL)
  {
    puts("decodeddataNULL!!! \n");
  }

  //cv::Mat rawData(256,sizeof(img_byte_size),CV_8SC1,(void*)img_bytes);
  //cv::Mat decodedMat= imdecode(rawData,cv::IMREAD_ANYDEPTH);

  //cv::Mat image = cv::Mat(245,245,CV_8UC3,img_bytes).clone();
  //imgbuf = cv::Mat(480, 640, CV_8U, img_bytes[img_byte_size]);
  //imgMat = cv::imdecode(imgbuf);
  // printf("n:%d %d\n", remaind ,evbuffer_get_length(buf));

  // //cv::imwrite("img.png",decodedImage);
  //
  //   cv::Mat image2=cv::imread("android_camera.png",cv::IMREAD_COLOR);
  // if(image2.empty())
  // {
  //       puts("data2 empty!!!!!!!!!!! \n");
  // }
  // cv::Mat image=cv::imread("android_camera2.png",cv::IMREAD_UNCHANGED);
  // if(image.empty())
  // {
  //       puts("data empty!!! \n");
  // }
  //  //cv::imshow("input",image);
  //  //cv::waitKey();
  //
  //cv::imwrite("img.png",decodedImage);

  static unsigned int count = 1;

  struct evbuffer *buf1 = evbuffer_new();
  if (!buf1)
  {
    puts("failed to create response buffer \n");
    return;
  }
  else
  {
    evbuffer_add_printf(buf1, "%s", test);
    //evhttp_send_reply(req, HTTP_OK, "OK", buf1);
    evhttp_send_reply(req, HTTP_OK, "OK", buf1);
    //evhttp_send_reply(req, 200, "OK", NULL);
    evbuffer_free(buf1);
  }

  printf("\n------------------%u------------------\n", count++);

  //RunMPPGraph(&decodedImage);
  picture_a = decodedImage;
}

void thread_callback()
{
  puts("thread callback!!");
  struct evhttp *httpd;
  event_init();
  httpd = evhttp_start("0.0.0.0", 5003);
  /* Set a callback for requests to "/specific". */
  /* evhttp_set_cb(httpd, "/specific", another_handler, NULL); */
  /* Set a callback for all other requests. */
  evhttp_set_gencb(httpd, HttpGenericCallback, NULL);

  event_dispatch(); /* Not reached in this code as it is now. */
  evhttp_free(httpd);
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::thread _t1(thread_callback);
  std::thread _t2(RunMPPGraph);
  //thread_callback();
  _t1.join();
  _t2.join();

  // ::mediapipe::Status run_status = RunMPPGraph();
  // if (!run_status.ok())
  // {
  //  LOG(ERROR) << "Failed to run the graph: " << run_status.message();
  //  return EXIT_FAILURE;
  // }
  // else
  // {
  //  LOG(INFO) << "Success!";
  // }
  return EXIT_SUCCESS;
}
