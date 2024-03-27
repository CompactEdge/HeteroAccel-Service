# HeteroAccel Service
## 이종가속기를 활용한 엣지 서비스 응용

### About
엣지 데이터 분업 처리를 위한 기계 학습 모델 실행 모듈은 이기종 아키텍처(CPU, GPU, FPGA)를 활용하는 AI 응용과 다양한 AI 데이터를 분업 처리하기 위한 기계학습 모델 및 알고리즘 실행 인터페이스를 지원

### Features
컨테이너화 된 응용 프로그램에 대한 데이터 및 관련 노드 정보들에 대한 메트릭을 가시화

* 이종가속기를 활용한 영상 추론 서비스
Human Pose, Gesture, Face, Text 에 대한 이미지를 추론하는 서비스를 제공

![image](https://github.com/CompactEdge/HeteroAccel-Service/assets/70622609/1a570495-4a4b-49b6-b14f-0aadcd2abca1)

* OpenVINO 기반 컴퓨팅 자원 실행 인터페이스
FPGA, GPU, NPU, CPU를 활용한 개별 이미지 추론 서비스에 대한 인터페이스를 설정하는 화면

![image](https://github.com/CompactEdge/HeteroAccel-Service/assets/70622609/1ca496fd-2948-4aa3-9dc8-0b411a5f6e50)

* 엣지 서비스 응용 상태 가시화
대시보드 기능을 통해 컴퓨팅 리소스와 응용의 상태를 확인

![image](https://github.com/CompactEdge/HeteroAccel-Service/assets/70622609/0803ebd3-3362-428f-bf01-ca559de9bc4a)
![image](https://github.com/CompactEdge/HeteroAccel-Service/assets/70622609/2008614b-2bb6-4e2f-895f-a21ad4bd015a)

### Architecture

![image](https://github.com/CompactEdge/HeteroAccel-Service/assets/70622609/98fdfb09-1ac2-4f0f-a890-c11fc4fdce27)
- 개발 언어: Python, C/C++ 등
- 개발 도구: Pycharm, Eclipse, Docker
- 운영체제: Linux (CentOS 8+), Kubetnetes
- CPU: Intel Xeon E 시리즈 / 스케일러블 시리즈
- GPU: INTEL 계열
- FPGA: Intel 또는 Xilinx
- VPU: Intel Molvidu VPUs
- 배포 플랫폼: 엣지 서버 플랫폼 

### Getting Started or Installation
1. 응용프로그램을  컨테이너화 하기 위해 Docker를 활용해 응용프로그램 이미지를 생성
2. 아래 이미지는 docker를 사용하여 컨테이터에 openvino를 설치하고 docker commit으로 이미지를 저장한 후 생성

![image](https://github.com/CompactEdge/HeteroAccel-Service/assets/70622609/ac6952cc-6269-4474-9138-314beacf0254)

3. intel gpu노드에 생성된 GPU 이미지임. 이미지를 생성한 후 컨테이너로 배포한다.

![image](https://github.com/CompactEdge/HeteroAccel-Service/assets/70622609/76a36178-e986-46df-a2b9-f221900af4ca)

4. repository 에서 제공하는 yaml을 활용해 서비스를 배포한다. yaml은 각 추론 기기 및 추론모델을 동작하는 명령어로 구성되어 있으며 외부와 통신을 하기 위해 kubernets service를 생성하여 각 컨테이너를 고유의 노드 포트와 매칭하여 외부에서 접근할 수 있도록 구성된다.

![image](https://github.com/CompactEdge/HeteroAccel-Service/assets/70622609/eed8a356-1ccf-44ef-a357-a99ddec399e0)
![image](https://github.com/CompactEdge/HeteroAccel-Service/assets/70622609/240e0006-9485-45ee-8def-9f31e389d9c4)

   
### TODO

